from dash import Dash, html, dcc
from flask import Response
import json
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import cv2
import os
import random
import threading

app = Dash(__name__)
server = app.server  # Flask server for custom routes

# ================================
# DATA LOADING AND IMAGE PROCESSING
# ================================

LABEL_NAMES = {
    0: 'T-shirt/Top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'
}

# Colour tints for each named colour option
COLOR_TINTS = {
    'White': (210, 210, 210),
    'Black': (35, 35, 35),
    'Gray': (130, 130, 130),
    'Charcoal': (70, 70, 70),
    'Navy': (31, 41, 95),
    'Cream': (225, 215, 185),
    'Ivory': (225, 220, 198),
    'Nude': (205, 168, 135),
    'Khaki': (182, 160, 115),
    'Camel': (188, 146, 86),
    'Tan': (192, 160, 115),
    'Beige': (208, 192, 162),
    'Olive': (95, 120, 55),
    'Sage': (135, 150, 108),
    'Burgundy': (115, 18, 35),
    'Red': (192, 38, 38),
    'Rust': (170, 58, 18),
    'Terracotta': (190, 90, 55),
    'Dusty Rose': (198, 135, 135),
    'Pink': (238, 160, 180),
    'Brown': (115, 70, 30),
    'Cognac': (140, 90, 35),
    'Linen': (188, 175, 155),
    'Light Blue': (148, 188, 218),
    'Flannel': (90, 95, 110),
    'Slate': (95, 110, 125),
}

# Initialise EDSR x4 super-resolution model once at startup
_SR_MODEL_PATH = os.path.join(os.path.dirname(__file__), "EDSR_x4.pb")
_sr = None
_sr_lock = threading.Lock()  # Thread lock for SR model
if os.path.exists(_SR_MODEL_PATH):
    try:
        _sr = cv2.dnn_superres.DnnSuperResImpl_create()
        _sr.readModel(_SR_MODEL_PATH)
        _sr.setModel("edsr", 4)
        print("Super-resolution model loaded (EDSR x4).")
    except Exception as _e:
        _sr = None
        print(f"Warning: could not load SR model ({_e}). Falling back to bicubic upscaling.")
else:
    print(f"Warning: {_SR_MODEL_PATH} not found. Falling back to bicubic upscaling.")


def load_data(filepath='product_images.csv'):
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} products")
        return df
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Creating sample data.")
        np.random.seed(42)
        n_samples = 100
        pixel_data = np.random.randint(0, 256, (n_samples, 784))
        labels = np.random.randint(0, 10, n_samples)
        columns = [f'pixel_{i}' for i in range(784)] + ['label']
        data = np.column_stack([pixel_data, labels])
        df = pd.DataFrame(data, columns=columns)
        return df


def pixels_to_base64(pixel_array, tint_rgb=None):
    """Convert 784 pixel values to a base64-encoded colour PNG using AI super-resolution."""
    img_array = np.array(pixel_array).reshape(28, 28).astype(np.uint8)
    img_array = 255 - img_array  # Invert: background=white, garment=dark

    if _sr is not None:
        with _sr_lock:  # Thread-safe access to SR model
            try:
                bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                upscaled = _sr.upsample(bgr)
                upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
            except cv2.error:
                upscaled = cv2.resize(img_array, (112, 112), interpolation=cv2.INTER_CUBIC)
    else:
        upscaled = cv2.resize(img_array, (112, 112), interpolation=cv2.INTER_CUBIC)

    if tint_rgb:
        r, g, b = tint_rgb
        gray_f = upscaled.astype(np.float32) / 255.0  # 0=dark garment, 1=white background
        garment_mask = 1.0 - gray_f                    # 1=garment, 0=background
        rgb_r = (255 * (1 - garment_mask) + r * garment_mask).clip(0, 255).astype(np.uint8)
        rgb_g = (255 * (1 - garment_mask) + g * garment_mask).clip(0, 255).astype(np.uint8)
        rgb_b = (255 * (1 - garment_mask) + b * garment_mask).clip(0, 255).astype(np.uint8)
        colored = np.stack([rgb_r, rgb_g, rgb_b], axis=2)
        img_out = Image.fromarray(colored, mode='RGB')
    else:
        img_out = Image.fromarray(upscaled, mode='L').convert('RGB')

    buffer = BytesIO()
    img_out.save(buffer, format='PNG')
    return buffer.getvalue()


def pixels_to_base64_str(pixel_array, tint_rgb=None):
    """Return base64 string version for backwards compatibility."""
    png_bytes = pixels_to_base64(pixel_array, tint_rgb)
    img_str = base64.b64encode(png_bytes).decode()
    return f"data:image/png;base64,{img_str}"


# Load data at startup
df = load_data()
pixel_cols = [f'pixel_{i}' for i in range(784)]

# In-memory cache for served images
image_cache = {}


def get_image_bytes(idx, tint_rgb=None):
    """Get image bytes with in-memory caching."""
    key = (idx, tint_rgb)
    if key not in image_cache:
        image_cache[key] = pixels_to_base64(df.iloc[idx][pixel_cols].values, tint_rgb)
    return image_cache[key]


# Flask route for lazy image loading
@server.route('/image/<int:idx>/<color>')
def serve_image(idx, color):
    """Serve product image on demand."""
    if idx < 0 or idx >= len(df):
        return Response(status=404)
    tint_rgb = COLOR_TINTS.get(color) if color != 'none' else None
    img_bytes = get_image_bytes(idx, tint_rgb)
    return Response(img_bytes, mimetype='image/png')


# ================================
# GENERATE DUMMY PRODUCT DATA
# ================================

random.seed(42)

DUMMY_PRODUCT_NAMES = {
    0: ['Essential Cotton Tee', 'Classic V-Neck Top', 'Basic Crew Neck', 'Slim Fit Tee', 'Relaxed Tank'],
    1: ['Slim Chino Pants', 'Wide Leg Trousers', 'Tailored Dress Pants', 'Casual Joggers', 'Linen Trousers'],
    2: ['Cashmere Pullover', 'Wool Blend Sweater', 'Cable Knit Jumper', 'Oversized Pullover', 'V-Neck Sweater'],
    3: ['Midi Wrap Dress', 'Linen Shift Dress', 'Evening Gown', 'Casual Sundress', 'Cocktail Dress'],
    4: ['Wool Overcoat', 'Trench Coat', 'Quilted Jacket', 'Pea Coat', 'Raincoat'],
    5: ['Leather Slides', 'Strappy Sandals', 'Platform Sandals', 'Gladiator Sandals', 'Espadrilles'],
    6: ['Oxford Button-Down', 'Silk Blouse', 'Linen Shirt', 'Flannel Shirt', 'Dress Shirt'],
    7: ['Minimal Sneaker', 'Canvas Low-Top', 'Running Shoes', 'High-Top Sneaker', 'Slip-On Sneaker'],
    8: ['Leather Tote', 'Crossbody Bag', 'Canvas Weekender', 'Clutch Bag', 'Backpack'],
    9: ['Chelsea Boots', 'Suede Ankle Boot', 'Heeled Boot', 'Combat Boots', 'Lace-Up Boots']
}

DUMMY_COLORS = {
    0: [['White', 'Black', 'Gray'], ['Navy', 'Cream'], ['Charcoal', 'White']],
    1: [['Khaki', 'Navy', 'Black'], ['Cream', 'Gray'], ['Olive', 'Tan']],
    2: [['Camel', 'Gray', 'Navy'], ['Cream', 'Charcoal'], ['Burgundy', 'Black']],
    3: [['Black', 'Sage', 'Rust'], ['White', 'Dusty Rose'], ['Navy', 'Burgundy']],
    4: [['Camel', 'Black', 'Gray'], ['Khaki', 'Navy'], ['Olive', 'Black']],
    5: [['Tan', 'Black'], ['Nude', 'White'], ['Brown', 'Navy']],
    6: [['White', 'Light Blue', 'Pink'], ['Ivory', 'Black'], ['Sage', 'Terracotta']],
    7: [['White', 'Black'], ['Navy', 'Gray'], ['Red', 'White']],
    8: [['Black', 'Tan', 'Burgundy'], ['Olive', 'Cognac'], ['Navy', 'Brown']],
    9: [['Black', 'Brown'], ['Tan', 'Gray'], ['Burgundy', 'Black']]
}

DUMMY_SIZES = {
    0: ['XS', 'S', 'M', 'L', 'XL'],
    1: ['28', '30', '32', '34', '36'],
    2: ['XS', 'S', 'M', 'L', 'XL'],
    3: ['XS', 'S', 'M', 'L'],
    4: ['XS', 'S', 'M', 'L', 'XL'],
    5: ['36', '37', '38', '39', '40', '41'],
    6: ['XS', 'S', 'M', 'L', 'XL'],
    7: ['36', '37', '38', '39', '40', '41', '42', '43', '44'],
    8: ['One Size'],
    9: ['36', '37', '38', '39', '40', '41', '42', '43', '44']
}

PRICE_RANGES = {
    0: (24, 49),
    1: (59, 99),
    2: (79, 149),
    3: (79, 149),
    4: (149, 299),
    5: (49, 99),
    6: (49, 129),
    7: (69, 169),
    8: (99, 199),
    9: (149, 229)
}


def generate_products(df, max_products=100):
    products = []
    name_counters = {i: 0 for i in range(10)}
    num_products = min(len(df), max_products)

    for idx in range(num_products):
        row = df.iloc[idx]
        label = int(row['label'])

        names = DUMMY_PRODUCT_NAMES[label]
        name = names[name_counters[label] % len(names)]
        name_counters[label] += 1

        min_price, max_price = PRICE_RANGES[label]
        price = random.randint(min_price, max_price)

        rating = round(random.uniform(3.5, 5.0), 1)
        reviews = random.randint(50, 500)

        colors = DUMMY_COLORS[label][idx % len(DUMMY_COLORS[label])]
        sizes = DUMMY_SIZES[label]

        # Store primary color for lazy image loading
        primary_color = colors[0] if colors else 'none'

        products.append({
            'id': f'p{idx}',
            'idx': idx,
            'name': f'{name} #{idx}',
            'category': label,
            'price': price,
            'rating': rating,
            'reviews': reviews,
            'colors': colors,
            'sizes': sizes,
            'primary_color': primary_color
        })

    return products


products = generate_products(df, max_products=20000)


def generate_fbt_pairs(products):
    fbt = {}
    category_map = {}

    for p in products:
        cat = p['category']
        if cat not in category_map:
            category_map[cat] = []
        category_map[cat].append(p['id'])

    complements = {
        0: [1, 7], 1: [0, 7], 2: [1, 9], 3: [5, 8],
        4: [2, 9], 5: [3, 8], 6: [1, 7], 7: [1, 0],
        8: [3, 4], 9: [1, 4]
    }

    for p in products:
        comp_cats = complements.get(p['category'], [])
        paired = []
        for comp_cat in comp_cats:
            if comp_cat in category_map and category_map[comp_cat]:
                paired.append(random.choice(category_map[comp_cat]))
        fbt[p['id']] = paired[:2]

    return fbt


fbt_pairs = generate_fbt_pairs(products)

products_json = json.dumps(products)
fbt_json = json.dumps(fbt_pairs)

initial_data = {
    "model_accuracy": "Not Connected Yet",
    "recommendations": [],
    "cart": [],
    "wishlist": []
}

# ================================
# HTML TEMPLATE
# ================================

HTML_CONTENT_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NUEM - Minimalist Fashion</title>
<script src="https://cdn.tailwindcss.com/3.4.17"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Playfair+Display:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  * { font-family: 'Space Grotesk', sans-serif; }
  .font-display { font-family: 'Playfair Display', serif; }
  .hide-scrollbar::-webkit-scrollbar { display: none; }
  .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
  @keyframes slideUp { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
  @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.25); } }
  .animate-slideUp { animation: slideUp 0.5s ease-out forwards; }
  .animate-fadeIn { animation: fadeIn 0.35s ease-out forwards; }
  .animate-pulse-once { animation: pulse 0.3s ease-out; }
  .product-card:hover .product-image { transform: scale(1.06); }
  .product-card:hover .quick-actions { opacity: 1; transform: translateY(0); }
  .quick-actions { opacity: 0; transform: translateY(12px); transition: all 0.3s ease; }
  .hero-section {
    background: linear-gradient(165deg, #0d0d0d 0%, #1c1c1c 40%, #111827 100%);
    position: relative;
    overflow: hidden;
  }
  .hero-section::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(120,100,80,0.15) 0%, transparent 60%);
  }
  input:focus { outline: none; }
  .card-shadow { box-shadow: 0 2px 12px rgba(0,0,0,0.07); }
  .card-shadow:hover { box-shadow: 0 6px 24px rgba(0,0,0,0.12); }
</style>
</head>

<body class="h-full bg-stone-50 text-stone-900">
<div id="app" class="h-full w-full overflow-auto">

<!-- ===== HEADER ===== -->
<header class="fixed top-0 left-0 right-0 bg-white/95 backdrop-blur-sm z-50 border-b border-stone-100">
  <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
    <button onclick="showHome()" class="text-2xl font-display font-semibold tracking-tight hover:opacity-70 transition">NUEM</button>
    <nav class="hidden md:flex items-center gap-6">
      <button onclick="showHome()" class="text-sm text-stone-500 hover:text-stone-900 transition font-medium">Home</button>
      <button onclick="showAllProducts()" class="text-sm text-stone-500 hover:text-stone-900 transition font-medium">Bestseller</button>
      <div class="relative group">
        <button class="text-sm text-stone-500 hover:text-stone-900 transition font-medium flex items-center gap-1">Categories <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg></button>
        <div class="absolute top-full left-0 mt-2 bg-white shadow-xl rounded-xl py-2 min-w-[160px] opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
          <button onclick="filterByCategory(0)" class="block w-full text-left px-4 py-2 text-sm text-stone-600 hover:bg-stone-50 hover:text-stone-900">T-shirt/Top</button>
          <button onclick="filterByCategory(1)" class="block w-full text-left px-4 py-2 text-sm text-stone-600 hover:bg-stone-50 hover:text-stone-900">Trouser</button>
          <button onclick="filterByCategory(2)" class="block w-full text-left px-4 py-2 text-sm text-stone-600 hover:bg-stone-50 hover:text-stone-900">Pullover</button>
          <button onclick="filterByCategory(3)" class="block w-full text-left px-4 py-2 text-sm text-stone-600 hover:bg-stone-50 hover:text-stone-900">Dress</button>
          <button onclick="filterByCategory(4)" class="block w-full text-left px-4 py-2 text-sm text-stone-600 hover:bg-stone-50 hover:text-stone-900">Coat</button>
          <button onclick="filterByCategory(5)" class="block w-full text-left px-4 py-2 text-sm text-stone-600 hover:bg-stone-50 hover:text-stone-900">Sandal</button>
          <button onclick="filterByCategory(6)" class="block w-full text-left px-4 py-2 text-sm text-stone-600 hover:bg-stone-50 hover:text-stone-900">Shirt</button>
          <button onclick="filterByCategory(7)" class="block w-full text-left px-4 py-2 text-sm text-stone-600 hover:bg-stone-50 hover:text-stone-900">Sneaker</button>
          <button onclick="filterByCategory(8)" class="block w-full text-left px-4 py-2 text-sm text-stone-600 hover:bg-stone-50 hover:text-stone-900">Bag</button>
          <button onclick="filterByCategory(9)" class="block w-full text-left px-4 py-2 text-sm text-stone-600 hover:bg-stone-50 hover:text-stone-900">Ankle Boot</button>
        </div>
      </div>
    </nav>
    <div class="flex items-center gap-2">
      <button onclick="showWishlist()" class="relative p-2 hover:bg-stone-100 rounded-full transition">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/>
        </svg>
        <span id="wishlistCount" class="absolute -top-1 -right-1 w-4 h-4 bg-stone-900 text-white text-xs rounded-full items-center justify-center hidden">0</span>
      </button>
      <button onclick="showCart()" class="relative p-2 hover:bg-stone-100 rounded-full transition">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z"/>
        </svg>
        <span id="cartCount" class="absolute -top-1 -right-1 w-4 h-4 bg-stone-900 text-white text-xs rounded-full items-center justify-center hidden">0</span>
      </button>
    </div>
  </div>
</header>

<!-- ===== MAIN CONTENT ===== -->
<main id="mainContent" class="pt-20">

  <!-- HOME VIEW -->
  <div id="homeView">
    <!-- Hero -->
    <section class="hero-section h-[30rem] flex items-center justify-center">
      <div class="relative text-center px-4 animate-slideUp z-10">
        <p class="text-stone-400 text-xs tracking-[0.3em] uppercase mb-5">New Collection</p>
        <h1 class="font-display text-6xl md:text-7xl font-medium mb-5 text-white leading-none">Less is More</h1>
        <p class="text-stone-400 text-base mb-10 max-w-sm mx-auto">Curated essentials for the modern wardrobe</p>
        <button onclick="showAllProducts()" class="bg-white text-stone-900 px-10 py-3.5 text-sm tracking-widest font-medium hover:bg-stone-100 transition">SHOP NOW</button>
      </div>
    </section>

    <!-- Categories -->
    <section class="max-w-7xl mx-auto px-6 py-16">
      <h2 class="font-display text-2xl mb-8 text-stone-900">Shop by Category</h2>
      <div id="categoriesGrid" class="grid grid-cols-2 md:grid-cols-5 gap-3"></div>
    </section>

    <!-- New Arrivals -->
    <section class="max-w-7xl mx-auto px-6 pb-16">
      <h2 class="font-display text-2xl mb-8 text-stone-900">New Arrivals</h2>
      <div id="featuredProducts" class="grid grid-cols-2 md:grid-cols-4 gap-5"></div>
    </section>
  </div>

  <!-- PRODUCTS VIEW -->
  <div id="productsView" class="hidden max-w-7xl mx-auto px-6 py-10">
    <div class="flex items-center justify-between mb-8">
      <h2 id="productsTitle" class="font-display text-2xl">All Products</h2>
      <select id="sortSelect" onchange="sortProducts()" class="border border-stone-200 px-4 py-2 text-sm bg-white rounded-lg focus:border-stone-400 transition">
        <option value="default">Sort by</option>
        <option value="price-low">Price: Low to High</option>
        <option value="price-high">Price: High to Low</option>
        <option value="rating">Top Rated</option>
      </select>
    </div>
    <div id="productsGrid" class="grid grid-cols-2 md:grid-cols-4 gap-5"></div>
  </div>

  <!-- PRODUCT DETAIL VIEW -->
  <div id="productDetailView" class="hidden max-w-7xl mx-auto px-6 py-10">
    <button onclick="goBack()" class="flex items-center gap-2 text-sm text-stone-500 hover:text-stone-900 mb-8 transition">
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
      </svg>
      Back
    </button>
    <div id="productDetail" class="grid md:grid-cols-2 gap-12"></div>
    <div id="recommendations" class="mt-16 space-y-14"></div>
  </div>

  <!-- CART VIEW -->
  <div id="cartView" class="hidden max-w-4xl mx-auto px-6 py-10">
    <h2 class="font-display text-2xl mb-8">Shopping Cart</h2>
    <div id="cartItems" class="space-y-4"></div>
    <div id="cartSummary" class="mt-8 border-t border-stone-200 pt-8"></div>
  </div>

  <!-- WISHLIST VIEW -->
  <div id="wishlistView" class="hidden max-w-7xl mx-auto px-6 py-10">
    <h2 class="font-display text-2xl mb-8">Wishlist</h2>
    <div id="wishlistItems" class="grid grid-cols-2 md:grid-cols-4 gap-5"></div>
  </div>

  <!-- ORDER CONFIRMATION VIEW -->
  <div id="confirmationView" class="hidden max-w-xl mx-auto px-6 py-20 text-center">
    <div class="w-20 h-20 bg-stone-900 rounded-full flex items-center justify-center mx-auto mb-8">
      <svg class="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
      </svg>
    </div>
    <h1 class="font-display text-4xl font-medium mb-4">Order Confirmed</h1>
    <p class="text-stone-500 mb-2">Thank you for shopping with NUEM.</p>
    <p class="text-stone-500 mb-8">A confirmation email has been sent to your inbox.</p>
    <p class="text-3xl font-semibold mb-10" id="confirmationTotal"></p>
    <button onclick="continueShopping()" class="bg-stone-900 text-white px-10 py-4 text-sm tracking-widest font-medium hover:bg-stone-800 transition">
      CONTINUE SHOPPING
    </button>
  </div>

</main>

<!-- ===== PAYMENT MODAL ===== -->
<div id="paymentModal" class="fixed inset-0 bg-black/60 z-50 hidden items-center justify-center p-4" onclick="closePaymentModalOutside(event)">
  <div class="bg-white max-w-md w-full rounded-2xl p-8 animate-fadeIn" id="paymentModalInner">
    <div class="flex items-center justify-between mb-6">
      <h2 class="font-display text-2xl">Payment Details</h2>
      <button onclick="closePaymentModal()" class="text-stone-400 hover:text-stone-900 transition">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
        </svg>
      </button>
    </div>

    <div id="paymentSummary" class="bg-stone-50 rounded-xl p-4 mb-6">
      <div class="flex justify-between text-sm text-stone-500 mb-1">
        <span>Subtotal</span><span id="modalSubtotal"></span>
      </div>
      <div class="flex justify-between text-sm text-stone-500 mb-2">
        <span>Shipping</span><span id="modalShipping"></span>
      </div>
      <div class="flex justify-between font-semibold text-lg pt-2 border-t border-stone-200">
        <span>Total</span><span id="modalTotal"></span>
      </div>
    </div>

    <div class="space-y-4">
      <div>
        <label class="block text-xs font-medium text-stone-600 mb-1.5 tracking-wide uppercase">Cardholder Name</label>
        <input type="text" id="cardName" placeholder="Jane Smith" class="w-full border border-stone-200 rounded-xl px-4 py-3 text-sm hover:border-stone-400 focus:border-stone-900 transition">
      </div>
      <div>
        <label class="block text-xs font-medium text-stone-600 mb-1.5 tracking-wide uppercase">Card Number</label>
        <input type="text" id="cardNumber" placeholder="1234 5678 9012 3456" maxlength="19" oninput="formatCardNumber(this)" class="w-full border border-stone-200 rounded-xl px-4 py-3 text-sm hover:border-stone-400 focus:border-stone-900 transition">
      </div>
      <div class="grid grid-cols-2 gap-4">
        <div>
          <label class="block text-xs font-medium text-stone-600 mb-1.5 tracking-wide uppercase">Expiry</label>
          <input type="text" id="cardExpiry" placeholder="MM / YY" maxlength="7" oninput="formatExpiry(this)" class="w-full border border-stone-200 rounded-xl px-4 py-3 text-sm hover:border-stone-400 focus:border-stone-900 transition">
        </div>
        <div>
          <label class="block text-xs font-medium text-stone-600 mb-1.5 tracking-wide uppercase">CVV</label>
          <input type="text" id="cardCvv" placeholder="123" maxlength="4" class="w-full border border-stone-200 rounded-xl px-4 py-3 text-sm hover:border-stone-400 focus:border-stone-900 transition">
        </div>
      </div>
    </div>

    <button onclick="processPayment()" class="w-full bg-stone-900 text-white py-4 rounded-xl font-medium text-sm tracking-widest mt-6 hover:bg-stone-800 transition">
      PAY NOW
    </button>
    <p class="text-xs text-stone-400 text-center mt-4">Your payment is simulated — no real charge is made.</p>
  </div>
</div>

<!-- ===== TOAST ===== -->
<div id="toast" class="fixed bottom-6 left-1/2 -translate-x-1/2 bg-stone-900 text-white px-6 py-3 rounded-full shadow-xl transform translate-y-20 opacity-0 transition-all duration-300 z-50 text-sm font-medium">
  <span id="toastMessage"></span>
</div>

<!-- SVG Defs -->
<svg class="hidden">
  <symbol id="star-full" viewBox="0 0 20 20">
    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
  </symbol>
</svg>

</div>

<script>
// ===== DATA =====
const products = __PRODUCTS_JSON__;
const frequentlyBoughtTogether = __FBT_JSON__;

const categories = [
  { id: 0, name: 'T-shirt/Top', icon: '👕' },
  { id: 1, name: 'Trouser',     icon: '👖' },
  { id: 2, name: 'Pullover',    icon: '🧥' },
  { id: 3, name: 'Dress',       icon: '👗' },
  { id: 4, name: 'Coat',        icon: '🧥' },
  { id: 5, name: 'Sandal',      icon: '🩴' },
  { id: 6, name: 'Shirt',       icon: '👔' },
  { id: 7, name: 'Sneaker',     icon: '👟' },
  { id: 8, name: 'Bag',         icon: '👜' },
  { id: 9, name: 'Ankle Boot',  icon: '🥾' }
];

// ===== STATE =====
let cart = [];
let wishlist = [];
let currentView = 'home';
let currentCategory = null;
let currentProduct = null;
let previousView = 'home';
let displayedProductCount = 20;
let currentProductList = [];
let isLoading = false;
const PRODUCTS_PER_LOAD = 20;

// ===== STORAGE =====
function saveCart()     { try { localStorage.setItem('nuem_cart',     JSON.stringify(cart));     } catch(e) {} }
function saveWishlist() { try { localStorage.setItem('nuem_wishlist', JSON.stringify(wishlist)); } catch(e) {} }
function loadStorage() {
  try {
    const c = localStorage.getItem('nuem_cart');     if (c) cart     = JSON.parse(c);
    const w = localStorage.getItem('nuem_wishlist'); if (w) wishlist = JSON.parse(w);
  } catch(e) {}
}

// ===== NAVIGATION =====
function showHome() {
  hideAllViews();
  document.getElementById('homeView').classList.remove('hidden');
  currentView = 'home';
}

function showAllProducts() {
  hideAllViews();
  document.getElementById('productsView').classList.remove('hidden');
  document.getElementById('productsTitle').textContent = 'All Products';
  currentCategory = null;
  currentView = 'products';
  currentProductList = [...products];
  displayedProductCount = PRODUCTS_PER_LOAD;
  renderProducts(currentProductList.slice(0, displayedProductCount), true);
}

function filterByCategory(categoryId) {
  hideAllViews();
  document.getElementById('productsView').classList.remove('hidden');
  currentCategory = categoryId;
  currentView = 'products';
  if (categoryId === null) {
    document.getElementById('productsTitle').textContent = 'All Products';
    currentProductList = [...products];
  } else {
    const category = categories.find(c => c.id === categoryId);
    document.getElementById('productsTitle').textContent = category.name;
    currentProductList = products.filter(p => p.category === categoryId);
  }
  displayedProductCount = PRODUCTS_PER_LOAD;
  renderProducts(currentProductList.slice(0, displayedProductCount), true);
}

function showProductDetail(productId) {
  previousView = currentView;
  hideAllViews();
  document.getElementById('productDetailView').classList.remove('hidden');
  currentView = 'productDetail';
  currentProduct = products.find(p => p.id === productId);
  renderProductDetail(currentProduct);
  renderRecommendations(currentProduct);
  document.getElementById('app').scrollTo({ top: 0, behavior: 'instant' });
}

function showCart() {
  previousView = currentView;
  hideAllViews();
  document.getElementById('cartView').classList.remove('hidden');
  currentView = 'cart';
  renderCart();
}

function showWishlist() {
  previousView = currentView;
  hideAllViews();
  document.getElementById('wishlistView').classList.remove('hidden');
  currentView = 'wishlist';
  renderWishlist();
}

function goBack() {
  if (previousView === 'home') showHome();
  else if (previousView === 'products') {
    if (currentCategory !== null) filterByCategory(currentCategory);
    else showAllProducts();
  }
  else if (previousView === 'cart') showCart();
  else if (previousView === 'wishlist') showWishlist();
  else showHome();
}

function hideAllViews() {
  ['homeView','productsView','productDetailView','cartView','wishlistView','confirmationView']
    .forEach(id => document.getElementById(id).classList.add('hidden'));
}

function continueShopping() {
  document.getElementById('confirmationView').classList.add('hidden');
  showHome();
}

// ===== RENDER FUNCTIONS =====
function renderCategories() {
  const grid = document.getElementById('categoriesGrid');
  grid.innerHTML = categories.map(cat => `
    <button onclick="filterByCategory(${cat.id})"
      class="bg-white p-5 rounded-xl hover:shadow-md transition group text-center card-shadow">
      <div class="text-3xl mb-2 group-hover:scale-110 transition">${cat.icon}</div>
      <div class="text-xs font-medium text-stone-700">${cat.name}</div>
    </button>
  `).join('');
}

function renderFeaturedProducts() {
  const container = document.getElementById('featuredProducts');
  const featured = products.filter(p => p.rating >= 4.5).slice(0, 8);
  container.innerHTML = featured.map(p => createProductCard(p)).join('');
}

function renderProducts(productList, reset = false) {
  const grid = document.getElementById('productsGrid');
  if (reset) {
    grid.innerHTML = productList.map(p => createProductCard(p)).join('');
  } else {
    grid.innerHTML += productList.map(p => createProductCard(p)).join('');
  }
  updateLoadMoreIndicator();
}

function loadMoreProducts() {
  if (isLoading || displayedProductCount >= currentProductList.length) return;
  isLoading = true;
  const nextBatch = currentProductList.slice(displayedProductCount, displayedProductCount + PRODUCTS_PER_LOAD);
  displayedProductCount += nextBatch.length;
  renderProducts(nextBatch, false);
  isLoading = false;
}

function updateLoadMoreIndicator() {
  let indicator = document.getElementById('loadMoreIndicator');
  const container = document.getElementById('productsView');
  if (!indicator) {
    indicator = document.createElement('div');
    indicator.id = 'loadMoreIndicator';
    indicator.className = 'text-center py-8 text-stone-400 text-sm';
    container.appendChild(indicator);
  }
  if (displayedProductCount >= currentProductList.length) {
    indicator.innerHTML = `<span class="text-stone-300">All ${currentProductList.length} products shown</span>`;
  } else {
    indicator.innerHTML = `<span>Showing ${displayedProductCount} of ${currentProductList.length} products</span><br><span class="text-xs">Scroll down for more</span>`;
  }
}

function renderStars(rating) {
  const full = Math.floor(rating);
  const half = rating % 1 >= 0.5;
  let stars = '';
  for (let i = 0; i < full; i++)
    stars += '<svg class="w-3 h-3 fill-current"><use href="#star-full"/></svg>';
  if (half)
    stars += '<svg class="w-3 h-3 fill-current opacity-40"><use href="#star-full"/></svg>';
  return stars;
}

function getImageUrl(product) {
  return `/image/${product.idx}/${product.primary_color || 'none'}`;
}

function createProductCard(product) {
  const isWishlisted = wishlist.some(w => w.product_id === product.id);
  const imageUrl = getImageUrl(product);

  return `
    <div class="product-card group cursor-pointer" onclick="showProductDetail('${product.id}')">
      <div class="relative aspect-square rounded-2xl overflow-hidden mb-3 bg-stone-100 card-shadow">
        <img src="${imageUrl}" loading="lazy" class="product-image w-full h-full object-contain transition duration-500" alt="${product.name}">
        <div class="quick-actions absolute bottom-0 left-0 right-0 p-3 flex gap-2 bg-gradient-to-t from-black/20 to-transparent">
          <button onclick="event.stopPropagation(); addToCart('${product.id}')"
            class="flex-1 bg-stone-900 text-white py-2 text-xs rounded-lg hover:bg-stone-700 transition font-medium">
            Add to Cart
          </button>
          <button onclick="event.stopPropagation(); toggleWishlist('${product.id}')"
            class="w-9 h-9 flex items-center justify-center bg-white rounded-lg hover:bg-stone-100 transition ${isWishlisted ? 'text-red-500' : 'text-stone-400'}">
            <svg class="w-4 h-4" fill="${isWishlisted ? 'currentColor' : 'none'}" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/>
            </svg>
          </button>
        </div>
      </div>
      <h3 class="text-sm font-medium mb-1 text-stone-800">${product.name}</h3>
      <div class="flex items-center gap-1.5 mb-1">
        <div class="flex items-center text-amber-500">${renderStars(product.rating)}</div>
        <span class="text-xs text-stone-400">(${product.reviews})</span>
      </div>
      <p class="font-semibold text-stone-900">£${product.price}</p>
    </div>
  `;
}

function renderProductDetail(product) {
  const isWishlisted = wishlist.some(w => w.product_id === product.id);
  const category = categories.find(c => c.id === product.category);
  const imageUrl = getImageUrl(product);

  document.getElementById('productDetail').innerHTML = `
    <div class="aspect-square rounded-2xl overflow-hidden bg-stone-100"><img src="${imageUrl}" class="w-full h-full object-contain" alt="${product.name}"></div>
    <div class="space-y-6">
      <div>
        <p class="text-xs text-stone-400 mb-2 tracking-wide uppercase">${category.name}</p>
        <h1 class="text-3xl font-display font-medium mb-3">${product.name}</h1>
        <div class="flex items-center gap-3 mb-5">
          <div class="flex items-center text-amber-500">${renderStars(product.rating)}</div>
          <span class="text-sm text-stone-400">${product.rating} · ${product.reviews} reviews</span>
        </div>
        <p class="text-3xl font-semibold">£${product.price}</p>
      </div>

      <div>
        <label class="block text-xs font-medium mb-2 tracking-wide uppercase text-stone-500">Colour</label>
        <div class="flex flex-wrap gap-2">
          ${product.colors.map((color, i) => `
            <button onclick="selectColor(this)"
              class="px-4 py-2 border ${i === 0 ? 'border-stone-900 bg-stone-50' : 'border-stone-200'} rounded-lg text-sm hover:border-stone-900 transition">
              ${color}
            </button>
          `).join('')}
        </div>
      </div>

      <div>
        <label class="block text-xs font-medium mb-2 tracking-wide uppercase text-stone-500">Size</label>
        <div class="flex flex-wrap gap-2">
          ${product.sizes.map((size, i) => `
            <button onclick="selectSize(this)"
              class="w-12 h-12 border ${i === 0 ? 'border-stone-900 bg-stone-50' : 'border-stone-200'} rounded-lg text-sm hover:border-stone-900 transition">
              ${size}
            </button>
          `).join('')}
        </div>
      </div>

      <div class="flex gap-3">
        <button onclick="addToCart('${product.id}')"
          class="flex-1 bg-stone-900 text-white py-4 rounded-xl font-medium text-sm hover:bg-stone-800 transition tracking-wide">
          Add to Cart
        </button>
        <button onclick="buyNow('${product.id}')"
          class="flex-1 border border-stone-900 py-4 rounded-xl font-medium text-sm hover:bg-stone-50 transition tracking-wide">
          Buy Now
        </button>
        <button onclick="toggleWishlist('${product.id}')"
          class="w-14 h-14 border ${isWishlisted ? 'border-red-400 text-red-500' : 'border-stone-200 text-stone-400'} rounded-xl flex items-center justify-center hover:border-stone-900 transition">
          <svg class="w-6 h-6" fill="${isWishlisted ? 'currentColor' : 'none'}" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
              d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/>
          </svg>
        </button>
      </div>

      <div class="border-t border-stone-100 pt-5 space-y-3">
        <div class="flex items-center gap-3 text-sm text-stone-500">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4"/>
          </svg>
          Free shipping on orders over £100
        </div>
        <div class="flex items-center gap-3 text-sm text-stone-500">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
          </svg>
          30-day easy returns
        </div>
      </div>
    </div>
  `;
}

function renderRecommendations(product) {
  const container = document.getElementById('recommendations');

  // Row 1: Similar Items (same category, excluding current)
  const similarItems = products
    .filter(p => p.category === product.category && p.id !== product.id)
    .slice(0, 4);

  // Row 2: Frequently Bought Together (as a grid), or cross-category picks if no FBT
  const fbtIds = frequentlyBoughtTogether[product.id] || [];
  const fbtProducts = fbtIds.map(id => products.find(p => p.id === id)).filter(Boolean);
  const youMayAlsoLike = fbtProducts.length >= 2
    ? fbtProducts
    : products.filter(p => p.id !== product.id && p.category !== product.category)
        .sort(() => Math.random() - 0.5)
        .slice(0, 4);

  container.innerHTML = `
    ${similarItems.length > 0 ? `
      <div>
        <h3 class="font-display text-xl mb-6">Similar Items</h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-5">
          ${similarItems.map(p => createProductCard(p)).join('')}
        </div>
      </div>
    ` : ''}

    ${youMayAlsoLike.length > 0 ? `
      <div>
        <h3 class="font-display text-xl mb-6">You May Also Like</h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-5">
          ${youMayAlsoLike.map(p => createProductCard(p)).join('')}
        </div>
      </div>
    ` : ''}
  `;
}

function renderCart() {
  const itemsContainer   = document.getElementById('cartItems');
  const summaryContainer = document.getElementById('cartSummary');

  if (cart.length === 0) {
    itemsContainer.innerHTML = `
      <div class="text-center py-16">
        <div class="text-6xl mb-5">🛒</div>
        <p class="text-stone-400 mb-6 text-lg">Your cart is empty</p>
        <button onclick="showAllProducts()"
          class="bg-stone-900 text-white px-8 py-3 rounded-xl text-sm font-medium hover:bg-stone-800 transition">
          Continue Shopping
        </button>
      </div>
    `;
    summaryContainer.innerHTML = '';
    return;
  }

  itemsContainer.innerHTML = cart.map(item => {
    const product = products.find(p => p.id === item.product_id);
    if (!product) return '';
    const imageUrl = getImageUrl(product);
    return `
      <div class="bg-white p-4 rounded-2xl flex gap-4 card-shadow">
        <img src="${imageUrl}" class="w-24 h-24 rounded-xl flex-shrink-0 object-contain bg-stone-100" alt="${product.name}">
        <div class="flex-1">
          <h3 class="font-medium mb-0.5">${product.name}</h3>
          <p class="text-sm text-stone-400 mb-3">£${product.price} each</p>
          <div class="flex items-center gap-2">
            <button onclick="updateQuantity('${item.__backendId}', ${item.quantity - 1})"
              class="w-8 h-8 border border-stone-200 rounded-lg flex items-center justify-center hover:bg-stone-100 transition text-stone-600">−</button>
            <span class="w-8 text-center font-medium">${item.quantity}</span>
            <button onclick="updateQuantity('${item.__backendId}', ${item.quantity + 1})"
              class="w-8 h-8 border border-stone-200 rounded-lg flex items-center justify-center hover:bg-stone-100 transition text-stone-600">+</button>
          </div>
        </div>
        <div class="flex flex-col items-end justify-between">
          <button onclick="removeFromCart('${item.__backendId}')" class="text-stone-300 hover:text-stone-600 transition">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
          <p class="font-semibold">£${(product.price * item.quantity).toFixed(2)}</p>
        </div>
      </div>
    `;
  }).join('');

  const subtotal = cart.reduce((sum, item) => {
    const product = products.find(p => p.id === item.product_id);
    return sum + (product ? product.price * item.quantity : 0);
  }, 0);
  const shipping = subtotal >= 100 ? 0 : 9.99;
  const total = subtotal + shipping;

  summaryContainer.innerHTML = `
    <div class="space-y-3 mb-6">
      <div class="flex justify-between text-stone-500">
        <span>Subtotal</span><span>£${subtotal.toFixed(2)}</span>
      </div>
      <div class="flex justify-between text-stone-500">
        <span>Shipping</span><span>${shipping === 0 ? 'Free' : '£' + shipping.toFixed(2)}</span>
      </div>
      <div class="flex justify-between text-lg font-semibold pt-3 border-t border-stone-200">
        <span>Total</span><span>£${total.toFixed(2)}</span>
      </div>
    </div>
    <button onclick="checkout()"
      class="w-full bg-stone-900 text-white py-4 rounded-xl font-medium text-sm tracking-widest hover:bg-stone-800 transition">
      CHECKOUT
    </button>
    <p class="text-xs text-stone-400 text-center mt-4">Free shipping on orders over £100</p>
  `;
}

function renderWishlist() {
  const container = document.getElementById('wishlistItems');

  if (wishlist.length === 0) {
    container.innerHTML = `
      <div class="col-span-full text-center py-16">
        <div class="text-6xl mb-5">💝</div>
        <p class="text-stone-400 mb-6 text-lg">Your wishlist is empty</p>
        <button onclick="showAllProducts()"
          class="bg-stone-900 text-white px-8 py-3 rounded-xl text-sm font-medium hover:bg-stone-800 transition">
          Explore Products
        </button>
      </div>
    `;
    return;
  }

  container.innerHTML = wishlist.map(item => {
    const product = products.find(p => p.id === item.product_id);
    if (!product) return '';
    return createProductCard(product);
  }).join('');
}

// ===== CART ACTIONS (local state, no SDK) =====
function addToCart(productId) {
  if (cart.length >= 999) { showToast('Cart is full'); return; }
  const existing = cart.find(c => c.product_id === productId);
  if (existing) {
    existing.quantity += 1;
    showToast('Quantity updated');
  } else {
    cart.push({
      product_id: productId,
      quantity: 1,
      __backendId: 'c_' + Date.now() + '_' + Math.random().toString(36).slice(2)
    });
    showToast('Added to cart');
  }
  updateCartBadge();
  animateCartBadge();
  saveCart();
}

function addBundle(mainId, otherIds) {
  const allIds = [mainId, ...otherIds];
  allIds.forEach(id => addToCart(id));
  showToast('Bundle added to cart');
}

function updateQuantity(backendId, newQty) {
  const idx = cart.findIndex(c => c.__backendId === backendId);
  if (idx === -1) return;
  if (newQty <= 0) {
    cart.splice(idx, 1);
    showToast('Item removed');
  } else {
    cart[idx].quantity = newQty;
  }
  updateCartBadge();
  saveCart();
  renderCart();
}

function removeFromCart(backendId) {
  const idx = cart.findIndex(c => c.__backendId === backendId);
  if (idx !== -1) {
    cart.splice(idx, 1);
    showToast('Item removed');
  }
  updateCartBadge();
  saveCart();
  renderCart();
}

// ===== WISHLIST ACTIONS =====
function toggleWishlist(productId) {
  const idx = wishlist.findIndex(w => w.product_id === productId);
  if (idx !== -1) {
    wishlist.splice(idx, 1);
    showToast('Removed from wishlist');
  } else {
    if (wishlist.length >= 999) { showToast('Wishlist is full'); return; }
    wishlist.push({
      product_id: productId,
      quantity: 1,
      __backendId: 'w_' + Date.now()
    });
    showToast('Added to wishlist');
    animateWishlistBadge();
  }
  updateWishlistBadge();
  saveWishlist();

  if (currentView === 'products') {
    if (currentCategory !== null) filterByCategory(currentCategory);
    else renderProducts(products);
  } else if (currentView === 'productDetail' && currentProduct) {
    renderProductDetail(currentProduct);
    renderRecommendations(currentProduct);
  } else if (currentView === 'wishlist') {
    renderWishlist();
  }
}

function buyNow(productId) {
  addToCart(productId);
  setTimeout(() => checkout(), 300);
}

// ===== CHECKOUT & PAYMENT =====
function checkout() {
  if (cart.length === 0) { showToast('Your cart is empty'); return; }

  const subtotal = cart.reduce((sum, item) => {
    const p = products.find(p => p.id === item.product_id);
    return sum + (p ? p.price * item.quantity : 0);
  }, 0);
  const shipping = subtotal >= 100 ? 0 : 9.99;
  const total = subtotal + shipping;

  document.getElementById('modalSubtotal').textContent = '£' + subtotal.toFixed(2);
  document.getElementById('modalShipping').textContent = shipping === 0 ? 'Free' : '£' + shipping.toFixed(2);
  document.getElementById('modalTotal').textContent = '£' + total.toFixed(2);

  const modal = document.getElementById('paymentModal');
  modal.classList.remove('hidden');
  modal.classList.add('flex');
}

function closePaymentModal() {
  const modal = document.getElementById('paymentModal');
  modal.classList.add('hidden');
  modal.classList.remove('flex');
}

function closePaymentModalOutside(e) {
  if (e.target === document.getElementById('paymentModal')) closePaymentModal();
}

function processPayment() {
  const subtotal = cart.reduce((sum, item) => {
    const p = products.find(p => p.id === item.product_id);
    return sum + (p ? p.price * item.quantity : 0);
  }, 0);
  const total = subtotal + (subtotal >= 100 ? 0 : 9.99);

  closePaymentModal();
  cart = [];
  saveCart();
  updateCartBadge();

  hideAllViews();
  document.getElementById('confirmationView').classList.remove('hidden');
  document.getElementById('confirmationTotal').textContent = '£' + total.toFixed(2);
  currentView = 'confirmation';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function formatCardNumber(input) {
  let val = input.value.replace(/\\D/g, '').slice(0, 16);
  input.value = val.replace(/(\\d{4})(?=\\d)/g, '$1 ');
}

function formatExpiry(input) {
  let val = input.value.replace(/\\D/g, '').slice(0, 4);
  if (val.length >= 3) val = val.slice(0, 2) + ' / ' + val.slice(2);
  input.value = val;
}

// ===== SORT =====
function sortProducts() {
  const sortBy = document.getElementById('sortSelect').value;
  let sorted = [...products];
  if (currentCategory !== null) sorted = sorted.filter(p => p.category === currentCategory);
  switch (sortBy) {
    case 'price-low':  sorted.sort((a, b) => a.price - b.price); break;
    case 'price-high': sorted.sort((a, b) => b.price - a.price); break;
    case 'rating':     sorted.sort((a, b) => b.rating - a.rating); break;
  }
  currentProductList = sorted;
  displayedProductCount = PRODUCTS_PER_LOAD;
  renderProducts(currentProductList.slice(0, displayedProductCount), true);
}

function selectColor(btn) {
  btn.parentElement.querySelectorAll('button').forEach(b => {
    b.classList.remove('border-stone-900', 'bg-stone-50');
    b.classList.add('border-stone-200');
  });
  btn.classList.remove('border-stone-200');
  btn.classList.add('border-stone-900', 'bg-stone-50');
}

function selectSize(btn) {
  btn.parentElement.querySelectorAll('button').forEach(b => {
    b.classList.remove('border-stone-900', 'bg-stone-50');
    b.classList.add('border-stone-200');
  });
  btn.classList.remove('border-stone-200');
  btn.classList.add('border-stone-900', 'bg-stone-50');
}

// ===== UI HELPERS =====
function updateCartBadge() {
  const badge = document.getElementById('cartCount');
  const count = cart.reduce((s, item) => s + item.quantity, 0);
  badge.textContent = count;
  badge.classList.toggle('hidden', count === 0);
  badge.classList.toggle('flex', count > 0);
}

function updateWishlistBadge() {
  const badge = document.getElementById('wishlistCount');
  badge.textContent = wishlist.length;
  badge.classList.toggle('hidden', wishlist.length === 0);
  badge.classList.toggle('flex', wishlist.length > 0);
}

function animateCartBadge() {
  const badge = document.getElementById('cartCount');
  badge.classList.add('animate-pulse-once');
  setTimeout(() => badge.classList.remove('animate-pulse-once'), 300);
}

function animateWishlistBadge() {
  const badge = document.getElementById('wishlistCount');
  badge.classList.add('animate-pulse-once');
  setTimeout(() => badge.classList.remove('animate-pulse-once'), 300);
}

function showToast(message) {
  const toast = document.getElementById('toast');
  document.getElementById('toastMessage').textContent = message;
  toast.classList.remove('translate-y-20', 'opacity-0');
  setTimeout(() => toast.classList.add('translate-y-20', 'opacity-0'), 2500);
}

// ===== INIT =====
loadStorage();
renderCategories();
renderFeaturedProducts();
updateCartBadge();
updateWishlistBadge();

// Infinite scroll listener
document.getElementById('app').addEventListener('scroll', function() {
  if (currentView !== 'products') return;
  const scrollTop = this.scrollTop;
  const scrollHeight = this.scrollHeight;
  const clientHeight = this.clientHeight;
  if (scrollTop + clientHeight >= scrollHeight - 300) {
    loadMoreProducts();
  }
});
</script>

</body>
</html>"""

# Inject product data
HTML_CONTENT = HTML_CONTENT_TEMPLATE \
    .replace('__PRODUCTS_JSON__', products_json) \
    .replace('__FBT_JSON__', fbt_json)

# ================================
# DASH LAYOUT
# ================================

app.layout = html.Div([
    dcc.Store(id="backend-data", data=initial_data),
    html.Iframe(
        srcDoc=HTML_CONTENT,
        style={"width": "100%", "height": "100vh", "border": "none"}
    )
])

# ================================
# RUN
# ================================

if __name__ == "__main__":
    app.run(debug=False)
