"""
Fashion E-Commerce Website using Dash
A modern, minimalist website to display clothing items from pixel data CSV
Styled after the NUEM template with cart, wishlist, and product detail pages
"""

import dash
from dash import html, dcc, callback, Input, Output, State, ctx, ALL, MATCH
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import math
import cv2
import os
import random

# ============================================================
# DATA LOADING AND IMAGE PROCESSING
# ============================================================

# Initialise EDSR x4 super-resolution model once at startup.
_SR_MODEL_PATH = os.path.join(os.path.dirname(__file__), "EDSR_x4.pb")
_sr = None
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

LABEL_NAMES = {
    0: 'T-shirt/Top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'
}

# Category icons (emojis)
CATEGORY_ICONS = {
    0: '👕', 1: '👖', 2: '🧥', 3: '👗', 4: '🧥',
    5: '🩴', 6: '👔', 7: '👟', 8: '👜', 9: '🥾'
}

# Generate dummy product data for each item
def generate_product_data(idx, label):
    """Generate dummy price, colors, sizes, rating, reviews for a product"""
    random.seed(idx)  # Consistent data per product
    
    # Price based on category
    base_prices = {0: 29, 1: 79, 2: 89, 3: 99, 4: 199, 5: 79, 6: 69, 7: 129, 8: 149, 9: 179}
    price = base_prices.get(label, 49) + random.randint(-10, 30)
    
    # Colors based on category
    color_options = {
        0: ['White', 'Black', 'Gray', 'Navy', 'Cream'],
        1: ['Khaki', 'Navy', 'Black', 'Gray', 'Cream'],
        2: ['Camel', 'Gray', 'Navy', 'Charcoal', 'Cream'],
        3: ['Black', 'Sage', 'Rust', 'Dusty Rose', 'Ivory'],
        4: ['Camel', 'Black', 'Gray', 'Khaki', 'Navy'],
        5: ['Tan', 'Black', 'Nude', 'Natural', 'Brown'],
        6: ['White', 'Light Blue', 'Pink', 'Ivory', 'Sage'],
        7: ['White', 'Black', 'Gray', 'Navy', 'Tan'],
        8: ['Black', 'Tan', 'Burgundy', 'Olive', 'Cognac'],
        9: ['Black', 'Brown', 'Tan', 'Gray', 'Burgundy']
    }
    colors = random.sample(color_options.get(label, ['Black', 'White']), min(3, len(color_options.get(label, []))))
    
    # Sizes based on category
    if label in [5, 7, 9]:  # Footwear
        sizes = ['36', '37', '38', '39', '40', '41', '42', '43', '44']
    elif label == 1:  # Trousers
        sizes = ['28', '30', '32', '34', '36']
    elif label == 8:  # Bags
        sizes = ['One Size']
    else:
        sizes = ['XS', 'S', 'M', 'L', 'XL']
    
    rating = round(3.5 + random.random() * 1.5, 1)
    reviews = random.randint(50, 500)
    
    return {
        'price': price,
        'colors': colors,
        'sizes': sizes,
        'rating': rating,
        'reviews': reviews
    }

# Category groupings for navigation
CATEGORIES = {
    'All': list(range(10)),
    'Tops': [0, 2, 4, 6],  # T-shirt, Pullover, Coat, Shirt
    'Bottoms': [1],  # Trouser
    'Dresses': [3],  # Dress
    'Footwear': [5, 7, 9],  # Sandal, Sneaker, Ankle Boot
    'Accessories': [8],  # Bag
}

def load_data(filepath='product_images.csv'):
    """Load product data from CSV"""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} products")
        return df
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Creating sample data.")
        # Create minimal sample data for testing
        np.random.seed(42)
        n_samples = 100
        pixel_data = np.random.randint(0, 256, (n_samples, 784))
        labels = np.random.randint(0, 10, n_samples)
        
        columns = [f'pixel_{i}' for i in range(784)] + ['label']
        data = np.column_stack([pixel_data, labels])
        df = pd.DataFrame(data, columns=columns)
        return df

def pixels_to_base64(pixel_array):
    """Convert 784 pixel values to a base64-encoded PNG using AI super-resolution."""
    img_array = np.array(pixel_array).reshape(28, 28).astype(np.uint8)
    # Invert: lower values = lighter in data, but we want dark items on light bg
    img_array = 255 - img_array

    if _sr is not None:
        try:
            # EDSR expects a 3-channel BGR image
            bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            upscaled = _sr.upsample(bgr)          # 28x28 -> 112x112 (x4)
            upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            upscaled = cv2.resize(img_array, (112, 112), interpolation=cv2.INTER_CUBIC)
    else:
        # Fallback: bicubic upscaling when model is unavailable
        upscaled = cv2.resize(img_array, (112, 112), interpolation=cv2.INTER_CUBIC)

    img = Image.fromarray(upscaled, mode='L')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Load data at startup
df = load_data()
pixel_cols = [f'pixel_{i}' for i in range(784)]

# Images are computed on first access and then cached (lazy loading).
image_cache = {}

def get_image(idx):
    if idx not in image_cache:
        image_cache[idx] = pixels_to_base64(df.iloc[idx][pixel_cols].values)
    return image_cache[idx]

# ============================================================
# DASH APP INITIALIZATION
# ============================================================

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "NUEM Fashion"

# ============================================================
# STYLES (Modern minimalist CSS inspired by NUEM template)
# ============================================================

# Color palette (stone colors)
COLORS = {
    'stone_50': '#fafaf9',
    'stone_100': '#f5f5f4',
    'stone_200': '#e7e5e4',
    'stone_300': '#d6d3d1',
    'stone_400': '#a8a29e',
    'stone_500': '#78716c',
    'stone_600': '#57534e',
    'stone_700': '#44403c',
    'stone_800': '#292524',
    'stone_900': '#1c1917',
    'amber_500': '#f59e0b',
    'red_500': '#ef4444',
}

STYLES = {
    'page': {
        'fontFamily': "'Space Grotesk', 'Arial', sans-serif",
        'backgroundColor': COLORS['stone_50'],
        'minHeight': '100vh',
        'color': COLORS['stone_900'],
    },
    'navbar': {
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'padding': '16px 24px',
        'borderBottom': f"1px solid {COLORS['stone_200']}",
        'backgroundColor': 'rgba(255, 255, 255, 0.95)',
        'backdropFilter': 'blur(8px)',
        'position': 'sticky',
        'top': 0,
        'zIndex': 1000,
    },
    'logo': {
        'fontSize': '24px',
        'fontFamily': "'Playfair Display', 'Georgia', serif",
        'fontWeight': '600',
        'letterSpacing': '1px',
        'color': COLORS['stone_900'],
        'textDecoration': 'none',
        'cursor': 'pointer',
    },
    'nav_links': {
        'display': 'flex',
        'gap': '32px',
        'listStyle': 'none',
        'margin': 0,
        'padding': 0,
    },
    'nav_link': {
        'textDecoration': 'none',
        'color': COLORS['stone_700'],
        'fontSize': '14px',
        'cursor': 'pointer',
        'padding': '8px 0',
        'transition': 'color 0.2s',
    },
    'icon_btn': {
        'position': 'relative',
        'padding': '8px',
        'borderRadius': '50%',
        'border': 'none',
        'backgroundColor': 'transparent',
        'cursor': 'pointer',
        'transition': 'background-color 0.2s',
    },
    'badge': {
        'position': 'absolute',
        'top': '-4px',
        'right': '-4px',
        'width': '20px',
        'height': '20px',
        'backgroundColor': COLORS['stone_900'],
        'color': 'white',
        'fontSize': '11px',
        'borderRadius': '50%',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
    },
    'hero': {
        'background': f"linear-gradient(135deg, {COLORS['stone_100']} 0%, {COLORS['stone_200']} 100%)",
        'padding': '96px 24px',
        'textAlign': 'center',
    },
    'hero_title': {
        'fontSize': '48px',
        'fontFamily': "'Playfair Display', 'Georgia', serif",
        'fontWeight': '500',
        'marginBottom': '16px',
        'color': COLORS['stone_900'],
    },
    'hero_subtitle': {
        'fontSize': '18px',
        'color': COLORS['stone_600'],
        'marginBottom': '32px',
    },
    'primary_btn': {
        'backgroundColor': COLORS['stone_900'],
        'color': 'white',
        'padding': '12px 32px',
        'border': 'none',
        'fontSize': '14px',
        'letterSpacing': '1px',
        'cursor': 'pointer',
        'transition': 'background-color 0.2s',
    },
    'secondary_btn': {
        'backgroundColor': 'white',
        'color': COLORS['stone_900'],
        'padding': '12px 32px',
        'border': f"1px solid {COLORS['stone_900']}",
        'fontSize': '14px',
        'cursor': 'pointer',
        'transition': 'background-color 0.2s',
    },
    'section': {
        'maxWidth': '1280px',
        'margin': '0 auto',
        'padding': '64px 24px',
    },
    'section_title': {
        'fontSize': '24px',
        'fontFamily': "'Playfair Display', 'Georgia', serif",
        'fontWeight': '500',
        'marginBottom': '32px',
        'color': COLORS['stone_900'],
    },
    'product_grid': {
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fill, minmax(250px, 1fr))',
        'gap': '24px',
    },
    'product_card': {
        'backgroundColor': 'white',
        'borderRadius': '8px',
        'overflow': 'hidden',
        'cursor': 'pointer',
        'transition': 'box-shadow 0.3s',
        'border': f"1px solid {COLORS['stone_200']}",
    },
    'product_image_container': {
        'position': 'relative',
        'aspectRatio': '1',
        'overflow': 'hidden',
        'backgroundColor': COLORS['stone_100'],
    },
    'product_image': {
        'width': '100%',
        'height': '100%',
        'objectFit': 'contain',
        'transition': 'transform 0.3s',
    },
    'quick_actions': {
        'position': 'absolute',
        'bottom': '12px',
        'left': '12px',
        'right': '12px',
        'display': 'flex',
        'gap': '8px',
        'opacity': 0,
        'transform': 'translateY(10px)',
        'transition': 'all 0.3s',
    },
    'product_info': {
        'padding': '12px',
    },
    'product_name': {
        'fontSize': '14px',
        'fontWeight': '500',
        'color': COLORS['stone_900'],
        'marginBottom': '4px',
    },
    'product_rating': {
        'display': 'flex',
        'alignItems': 'center',
        'gap': '8px',
        'marginBottom': '4px',
    },
    'stars': {
        'color': COLORS['amber_500'],
        'fontSize': '12px',
    },
    'review_count': {
        'fontSize': '12px',
        'color': COLORS['stone_500'],
    },
    'product_price': {
        'fontSize': '16px',
        'fontWeight': '600',
        'color': COLORS['stone_900'],
    },
    'category_grid': {
        'display': 'grid',
        'gridTemplateColumns': 'repeat(5, 1fr)',
        'gap': '16px',
    },
    'category_card': {
        'backgroundColor': 'white',
        'padding': '24px',
        'borderRadius': '8px',
        'textAlign': 'center',
        'cursor': 'pointer',
        'transition': 'box-shadow 0.2s',
        'border': f"1px solid {COLORS['stone_200']}",
    },
    'category_icon': {
        'fontSize': '32px',
        'marginBottom': '8px',
    },
    'category_name': {
        'fontSize': '14px',
        'fontWeight': '500',
        'color': COLORS['stone_900'],
    },
    'breadcrumb': {
        'padding': '16px 24px',
        'fontSize': '14px',
        'color': COLORS['stone_500'],
        'maxWidth': '1280px',
        'margin': '0 auto',
    },
    'pagination': {
        'display': 'flex',
        'justifyContent': 'center',
        'gap': '8px',
        'padding': '40px 0',
    },
    'page_btn': {
        'padding': '10px 16px',
        'border': f"1px solid {COLORS['stone_300']}",
        'backgroundColor': 'white',
        'borderRadius': '4px',
        'cursor': 'pointer',
        'fontSize': '14px',
        'transition': 'all 0.2s',
    },
    'page_btn_active': {
        'backgroundColor': COLORS['stone_900'],
        'color': 'white',
        'border': f"1px solid {COLORS['stone_900']}",
    },
    'filter_bar': {
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'padding': '16px 24px',
        'maxWidth': '1280px',
        'margin': '0 auto',
    },
    'results_count': {
        'fontSize': '14px',
        'color': COLORS['stone_600'],
    },
    'footer': {
        'backgroundColor': COLORS['stone_100'],
        'padding': '64px 24px',
        'borderTop': f"1px solid {COLORS['stone_200']}",
        'marginTop': '64px',
    },
    'footer_grid': {
        'display': 'grid',
        'gridTemplateColumns': 'repeat(4, 1fr)',
        'gap': '48px',
        'maxWidth': '1280px',
        'margin': '0 auto',
    },
    'footer_title': {
        'fontSize': '12px',
        'fontWeight': '600',
        'textTransform': 'uppercase',
        'letterSpacing': '1px',
        'marginBottom': '16px',
        'color': COLORS['stone_900'],
    },
    'footer_link': {
        'fontSize': '14px',
        'color': COLORS['stone_600'],
        'textDecoration': 'none',
        'display': 'block',
        'marginBottom': '8px',
        'cursor': 'pointer',
    },
    'footer_bottom': {
        'textAlign': 'center',
        'paddingTop': '32px',
        'marginTop': '32px',
        'borderTop': f"1px solid {COLORS['stone_200']}",
        'fontSize': '12px',
        'color': COLORS['stone_500'],
    },
    # Product Detail Page Styles
    'detail_container': {
        'maxWidth': '1280px',
        'margin': '0 auto',
        'padding': '32px 24px',
    },
    'detail_grid': {
        'display': 'grid',
        'gridTemplateColumns': '1fr 1fr',
        'gap': '48px',
    },
    'detail_image': {
        'width': '100%',
        'aspectRatio': '1',
        'objectFit': 'contain',
        'backgroundColor': COLORS['stone_100'],
        'borderRadius': '8px',
    },
    'detail_info': {
        'display': 'flex',
        'flexDirection': 'column',
        'gap': '24px',
    },
    'detail_category': {
        'fontSize': '14px',
        'color': COLORS['stone_500'],
        'marginBottom': '8px',
    },
    'detail_title': {
        'fontSize': '32px',
        'fontFamily': "'Playfair Display', 'Georgia', serif",
        'fontWeight': '500',
        'marginBottom': '8px',
    },
    'detail_price': {
        'fontSize': '28px',
        'fontWeight': '600',
    },
    'option_label': {
        'fontSize': '14px',
        'fontWeight': '500',
        'marginBottom': '8px',
    },
    'option_buttons': {
        'display': 'flex',
        'gap': '8px',
        'flexWrap': 'wrap',
    },
    'option_btn': {
        'padding': '8px 16px',
        'border': f"1px solid {COLORS['stone_300']}",
        'backgroundColor': 'white',
        'borderRadius': '4px',
        'cursor': 'pointer',
        'fontSize': '14px',
        'transition': 'all 0.2s',
    },
    'option_btn_active': {
        'border': f"1px solid {COLORS['stone_900']}",
    },
    'size_btn': {
        'width': '48px',
        'height': '48px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'border': f"1px solid {COLORS['stone_300']}",
        'backgroundColor': 'white',
        'borderRadius': '4px',
        'cursor': 'pointer',
        'fontSize': '14px',
        'transition': 'all 0.2s',
    },
    'action_buttons': {
        'display': 'flex',
        'gap': '12px',
    },
    'shipping_info': {
        'borderTop': f"1px solid {COLORS['stone_200']}",
        'paddingTop': '24px',
        'display': 'flex',
        'flexDirection': 'column',
        'gap': '12px',
    },
    'shipping_item': {
        'display': 'flex',
        'alignItems': 'center',
        'gap': '12px',
        'fontSize': '14px',
        'color': COLORS['stone_600'],
    },
    # Cart Styles
    'cart_container': {
        'maxWidth': '896px',
        'margin': '0 auto',
        'padding': '32px 24px',
    },
    'cart_item': {
        'display': 'flex',
        'gap': '16px',
        'padding': '16px',
        'backgroundColor': 'white',
        'borderRadius': '8px',
        'marginBottom': '16px',
        'border': f"1px solid {COLORS['stone_200']}",
    },
    'cart_item_image': {
        'width': '96px',
        'height': '96px',
        'borderRadius': '8px',
        'objectFit': 'contain',
        'backgroundColor': COLORS['stone_100'],
    },
    'cart_item_info': {
        'flex': 1,
    },
    'quantity_controls': {
        'display': 'flex',
        'alignItems': 'center',
        'gap': '12px',
    },
    'quantity_btn': {
        'width': '32px',
        'height': '32px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'border': f"1px solid {COLORS['stone_300']}",
        'backgroundColor': 'white',
        'borderRadius': '4px',
        'cursor': 'pointer',
        'fontSize': '16px',
    },
    'cart_summary': {
        'borderTop': f"1px solid {COLORS['stone_200']}",
        'paddingTop': '32px',
        'marginTop': '32px',
    },
    'summary_row': {
        'display': 'flex',
        'justifyContent': 'space-between',
        'marginBottom': '12px',
        'fontSize': '14px',
    },
    'summary_total': {
        'display': 'flex',
        'justifyContent': 'space-between',
        'paddingTop': '16px',
        'borderTop': f"1px solid {COLORS['stone_200']}",
        'fontSize': '18px',
        'fontWeight': '600',
    },
    # Toast Notification
    'toast': {
        'position': 'fixed',
        'bottom': '24px',
        'left': '50%',
        'transform': 'translateX(-50%)',
        'backgroundColor': COLORS['stone_900'],
        'color': 'white',
        'padding': '12px 24px',
        'borderRadius': '8px',
        'boxShadow': '0 4px 12px rgba(0,0,0,0.15)',
        'zIndex': 2000,
        'transition': 'all 0.3s',
    },
    # Recommendations
    'recommendations_section': {
        'marginTop': '64px',
    },
    'recommendations_title': {
        'fontSize': '20px',
        'fontFamily': "'Playfair Display', 'Georgia', serif",
        'marginBottom': '24px',
    },
    'horizontal_scroll': {
        'display': 'flex',
        'gap': '16px',
        'overflowX': 'auto',
        'paddingBottom': '16px',
    },
    # Wishlist heart button
    'heart_btn': {
        'width': '40px',
        'height': '40px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'backgroundColor': 'white',
        'border': f"1px solid {COLORS['stone_300']}",
        'borderRadius': '4px',
        'cursor': 'pointer',
        'transition': 'all 0.2s',
    },
    'heart_btn_active': {
        'color': COLORS['red_500'],
        'borderColor': COLORS['red_500'],
    },
    # Empty state
    'empty_state': {
        'textAlign': 'center',
        'padding': '96px 24px',
    },
    'empty_icon': {
        'fontSize': '64px',
        'marginBottom': '16px',
    },
    'empty_text': {
        'color': COLORS['stone_500'],
        'marginBottom': '24px',
    },
}

# ============================================================
# COMPONENTS
# ============================================================

def create_navbar():
    """Create modern navigation bar with cart and wishlist icons"""
    return html.Header([
        html.Div([
            html.Button("NUEM", style=STYLES['logo'], id='logo-home'),
            html.Nav([
                html.Button("Home", id='nav-home', style=STYLES['nav_link']),
                html.Button("Shop All", id='nav-shop', style=STYLES['nav_link']),
                *[html.Button(cat, id={'type': 'nav-category', 'category': cat}, style=STYLES['nav_link']) 
                  for cat in ['Tops', 'Bottoms', 'Dresses', 'Footwear', 'Accessories']],
            ], style={'display': 'flex', 'gap': '32px', 'alignItems': 'center'}),
            html.Div([
                # Wishlist button
                html.Button([
                    html.Div([
                        # Heart SVG
                        html.Div("♡", style={'fontSize': '20px'}),
                        html.Span(id='wishlist-badge', style={**STYLES['badge'], 'display': 'none'}),
                    ], style={'position': 'relative'})
                ], id='show-wishlist-btn', style=STYLES['icon_btn']),
                # Cart button
                html.Button([
                    html.Div([
                        # Cart SVG
                        html.Div("🛒", style={'fontSize': '18px'}),
                        html.Span(id='cart-badge', style={**STYLES['badge'], 'display': 'none'}),
                    ], style={'position': 'relative'})
                ], id='show-cart-btn', style=STYLES['icon_btn']),
            ], style={'display': 'flex', 'gap': '8px'}),
        ], style=STYLES['navbar'])
    ])

def create_hero():
    """Create hero banner section"""
    return html.Section([
        html.Div([
            html.H1("Less is More", style=STYLES['hero_title']),
            html.P("Curated essentials for the modern wardrobe", style=STYLES['hero_subtitle']),
            html.Button("SHOP NOW", id='hero-shop-btn', style=STYLES['primary_btn']),
        ], style={'animation': 'slideUp 0.4s ease-out'})
    ], style=STYLES['hero'])

def create_category_grid():
    """Create category selection grid with icons"""
    return html.Div([
        html.Div([
            html.Div(CATEGORY_ICONS[label_id], style=STYLES['category_icon']),
            html.Div(LABEL_NAMES[label_id], style=STYLES['category_name']),
        ], style=STYLES['category_card'], id={'type': 'category-card', 'label': label_id})
        for label_id in range(10)
    ], style=STYLES['category_grid'])

def create_product_card(idx, label):
    """Create a modern product card with quick actions"""
    product_data = generate_product_data(idx, label)
    
    return html.Div([
        # Image container with quick actions
        html.Div([
            html.Img(
                src=get_image(idx),
                style=STYLES['product_image'],
            ),
            # Quick action buttons (shown on hover via CSS)
            html.Div([
                html.Button("Add to Cart", 
                    id={'type': 'quick-add-cart', 'index': idx},
                    style={**STYLES['primary_btn'], 'flex': 1, 'padding': '8px 12px', 'fontSize': '12px'}),
                html.Button("♡",
                    id={'type': 'quick-wishlist', 'index': idx},
                    style={**STYLES['heart_btn'], 'fontSize': '16px'}),
            ], style=STYLES['quick_actions'], className='quick-actions'),
        ], style=STYLES['product_image_container'], className='product-image-container'),
        # Product info
        html.Div([
            html.Div(f"Product #{idx}", style=STYLES['product_name']),
            html.Div([
                html.Span("★" * int(product_data['rating']) + "☆" * (5 - int(product_data['rating'])), 
                         style=STYLES['stars']),
                html.Span(f"({product_data['reviews']})", style=STYLES['review_count']),
            ], style=STYLES['product_rating']),
            html.Div(f"${product_data['price']}", style=STYLES['product_price']),
        ], style=STYLES['product_info']),
    ], style=STYLES['product_card'], id={'type': 'product-card', 'index': idx}, className='product-card')

def create_footer():
    """Create footer section"""
    return html.Footer([
        html.Div([
            html.Div([
                html.Div("ABOUT", style=STYLES['footer_title']),
                html.A("Our Story", style=STYLES['footer_link']),
                html.A("Careers", style=STYLES['footer_link']),
                html.A("Sustainability", style=STYLES['footer_link']),
            ]),
            html.Div([
                html.Div("CUSTOMER SERVICE", style=STYLES['footer_title']),
                html.A("Contact Us", style=STYLES['footer_link']),
                html.A("FAQs", style=STYLES['footer_link']),
                html.A("Size Guide", style=STYLES['footer_link']),
            ]),
            html.Div([
                html.Div("ORDERS", style=STYLES['footer_title']),
                html.A("Shipping", style=STYLES['footer_link']),
                html.A("Returns", style=STYLES['footer_link']),
                html.A("Track Order", style=STYLES['footer_link']),
            ]),
            html.Div([
                html.Div("LEGAL", style=STYLES['footer_title']),
                html.A("Terms & Conditions", style=STYLES['footer_link']),
                html.A("Privacy Policy", style=STYLES['footer_link']),
                html.A("Cookie Policy", style=STYLES['footer_link']),
            ]),
        ], style=STYLES['footer_grid']),
        html.Div("© 2026 NUEM Fashion. All rights reserved.", style=STYLES['footer_bottom']),
    ], style=STYLES['footer'])

def create_breadcrumb(category='All', product_name=None):
    """Create breadcrumb navigation"""
    items = [
        html.Span("Home", style={**STYLES['nav_link'], 'cursor': 'pointer'}, id='breadcrumb-home'),
    ]
    if category != 'All':
        items.append(html.Span(" / ", style={'margin': '0 8px'}))
        items.append(html.Span(category, style={'color': COLORS['stone_700']} if not product_name else {**STYLES['nav_link'], 'cursor': 'pointer'}))
    if product_name:
        items.append(html.Span(" / ", style={'margin': '0 8px'}))
        items.append(html.Span(product_name, style={'color': COLORS['stone_700']}))
    return html.Div(items, style=STYLES['breadcrumb'])

def create_pagination(current_page, total_pages):
    """Create pagination controls"""
    buttons = []
    
    # Previous button
    buttons.append(html.Button(
        "← Prev",
        id={'type': 'page-btn', 'page': max(1, current_page - 1)},
        style={**STYLES['page_btn'], 'opacity': '0.5' if current_page == 1 else '1'},
        disabled=current_page == 1,
    ))
    
    # Page numbers
    start_page = max(1, current_page - 2)
    end_page = min(total_pages, current_page + 2)
    
    if start_page > 1:
        buttons.append(html.Button("1", id={'type': 'page-btn', 'page': 1}, style=STYLES['page_btn']))
        if start_page > 2:
            buttons.append(html.Span("...", style={'padding': '10px'}))
    
    for page in range(start_page, end_page + 1):
        style = {**STYLES['page_btn'], **STYLES['page_btn_active']} if page == current_page else STYLES['page_btn']
        buttons.append(html.Button(str(page), id={'type': 'page-btn', 'page': page}, style=style))
    
    if end_page < total_pages:
        if end_page < total_pages - 1:
            buttons.append(html.Span("...", style={'padding': '10px'}))
        buttons.append(html.Button(str(total_pages), id={'type': 'page-btn', 'page': total_pages}, style=STYLES['page_btn']))
    
    # Next button
    buttons.append(html.Button(
        "Next →",
        id={'type': 'page-btn', 'page': min(total_pages, current_page + 1)},
        style={**STYLES['page_btn'], 'opacity': '0.5' if current_page == total_pages else '1'},
        disabled=current_page == total_pages,
    ))
    
    return html.Div(buttons, style=STYLES['pagination'])

def create_product_detail(idx, label, wishlist):
    """Create product detail page content"""
    product_data = generate_product_data(idx, label)
    is_wishlisted = idx in wishlist
    
    return html.Div([
        # Back button
        html.Button([
            html.Span("←", style={'marginRight': '8px'}),
            "Back"
        ], id='back-btn', style={**STYLES['nav_link'], 'marginBottom': '32px', 'display': 'flex', 'alignItems': 'center'}),
        
        # Product grid
        html.Div([
            # Image
            html.Img(src=get_image(idx), style=STYLES['detail_image']),
            
            # Info
            html.Div([
                html.P(LABEL_NAMES[label], style=STYLES['detail_category']),
                html.H1(f"Product #{idx}", style=STYLES['detail_title']),
                html.Div([
                    html.Span("★" * int(product_data['rating']) + "☆" * (5 - int(product_data['rating'])), 
                             style={**STYLES['stars'], 'fontSize': '16px'}),
                    html.Span(f" {product_data['rating']} ({product_data['reviews']} reviews)", 
                             style={'color': COLORS['stone_500'], 'marginLeft': '8px'}),
                ], style={'marginBottom': '16px'}),
                html.P(f"${product_data['price']}", style=STYLES['detail_price']),
                
                # Color selection
                html.Div([
                    html.Label("Color", style=STYLES['option_label']),
                    html.Div([
                        html.Button(color, 
                            id={'type': 'color-option', 'color': color, 'index': idx},
                            style={**STYLES['option_btn'], **(STYLES['option_btn_active'] if i == 0 else {})})
                        for i, color in enumerate(product_data['colors'])
                    ], style=STYLES['option_buttons']),
                ], style={'marginTop': '24px'}),
                
                # Size selection
                html.Div([
                    html.Label("Size", style=STYLES['option_label']),
                    html.Div([
                        html.Button(size,
                            id={'type': 'size-option', 'size': size, 'index': idx},
                            style={**STYLES['size_btn'], **(STYLES['option_btn_active'] if i == 0 else {})})
                        for i, size in enumerate(product_data['sizes'])
                    ], style=STYLES['option_buttons']),
                ], style={'marginTop': '24px'}),
                
                # Action buttons
                html.Div([
                    html.Button("Add to Cart", 
                        id={'type': 'detail-add-cart', 'index': idx},
                        style={**STYLES['primary_btn'], 'flex': 1, 'padding': '16px'}),
                    html.Button("Buy Now",
                        id={'type': 'buy-now', 'index': idx},
                        style={**STYLES['secondary_btn'], 'flex': 1, 'padding': '16px'}),
                    html.Button("♥" if is_wishlisted else "♡",
                        id={'type': 'detail-wishlist', 'index': idx},
                        style={**STYLES['heart_btn'], 'width': '56px', 'height': '56px', 'fontSize': '24px',
                               **(STYLES['heart_btn_active'] if is_wishlisted else {})}),
                ], style=STYLES['action_buttons']),
                
                # Shipping info
                html.Div([
                    html.Div([
                        html.Span("📦", style={'fontSize': '20px'}),
                        html.Span("Free shipping on orders over $100"),
                    ], style=STYLES['shipping_item']),
                    html.Div([
                        html.Span("🔄", style={'fontSize': '20px'}),
                        html.Span("30-day easy returns"),
                    ], style=STYLES['shipping_item']),
                ], style=STYLES['shipping_info']),
            ], style=STYLES['detail_info']),
        ], style=STYLES['detail_grid']),
        
        # Recommendations
        create_recommendations(idx, label),
    ], style=STYLES['detail_container'])

def create_recommendations(idx, label):
    """Create product recommendations section"""
    # Get similar items (same category)
    similar_indices = df[df['label'] == label].index.tolist()
    similar_indices = [i for i in similar_indices if i != idx][:8]
    
    # Get top rated items
    all_indices = df.index.tolist()
    random.seed(idx + 100)
    top_rated = random.sample([i for i in all_indices if i != idx], min(8, len(all_indices) - 1))
    
    # Get frequently bought together (random items from different categories for demo)
    random.seed(idx)
    other_labels = [l for l in range(10) if l != label]
    fbt_indices = []
    for ol in random.sample(other_labels, min(2, len(other_labels))):
        others = df[df['label'] == ol].index.tolist()
        if others:
            fbt_indices.append(random.choice(others))
    
    sections = []
    
    # Frequently bought together
    if fbt_indices:
        current_price = generate_product_data(idx, label)['price']
        fbt_prices = [generate_product_data(i, df.iloc[i]['label'])['price'] for i in fbt_indices]
        bundle_price = current_price + sum(fbt_prices)
        
        sections.append(html.Div([
            html.H3("Frequently Bought Together", style=STYLES['recommendations_title']),
            html.Div([
                # Current product
                html.Div([
                    html.Img(src=get_image(idx), style={'width': '128px', 'height': '128px', 'objectFit': 'contain', 
                                                         'backgroundColor': COLORS['stone_100'], 'borderRadius': '8px'}),
                    html.P(f"Product #{idx}", style={'fontSize': '12px', 'marginTop': '8px'}),
                    html.P(f"${current_price}", style={'fontSize': '14px', 'fontWeight': '600'}),
                ], style={'textAlign': 'center', 'minWidth': '140px'}),
                *[
                    html.Div([
                        html.Span("+", style={'fontSize': '24px', 'color': COLORS['stone_300'], 'padding': '0 16px'}),
                        html.Div([
                            html.Img(src=get_image(fi), style={'width': '128px', 'height': '128px', 'objectFit': 'contain',
                                                                'backgroundColor': COLORS['stone_100'], 'borderRadius': '8px', 'cursor': 'pointer'},
                                    id={'type': 'fbt-item', 'index': fi}),
                            html.P(f"Product #{fi}", style={'fontSize': '12px', 'marginTop': '8px'}),
                            html.P(f"${fbt_prices[j]}", style={'fontSize': '14px', 'fontWeight': '600'}),
                        ], style={'textAlign': 'center', 'minWidth': '140px'})
                    ], style={'display': 'flex', 'alignItems': 'center'})
                    for j, fi in enumerate(fbt_indices)
                ],
                # Bundle price box
                html.Div([
                    html.P("Bundle Price", style={'fontSize': '14px', 'color': COLORS['stone_600'], 'marginBottom': '8px'}),
                    html.P(f"${bundle_price}", style={'fontSize': '20px', 'fontWeight': '600', 'marginBottom': '12px'}),
                    html.Button("Add All to Cart", 
                        id={'type': 'add-bundle', 'indices': [idx] + fbt_indices},
                        style={**STYLES['primary_btn'], 'padding': '8px 16px', 'fontSize': '12px'}),
                ], style={'backgroundColor': COLORS['stone_100'], 'padding': '16px', 'borderRadius': '8px', 
                          'marginLeft': '24px', 'minWidth': '150px'}),
            ], style={**STYLES['horizontal_scroll'], 'alignItems': 'center'}),
        ], style={'marginBottom': '48px'}))
    
    # 4 Stars and Above
    sections.append(html.Div([
        html.H3("4 Stars and Above ⭐", style=STYLES['recommendations_title']),
        html.Div([
            create_mini_product_card(i, df.iloc[i]['label'])
            for i in top_rated[:4]
        ], style=STYLES['product_grid']),
    ], style={'marginBottom': '48px'}))
    
    # Related products
    if similar_indices:
        sections.append(html.Div([
            html.H3("Products Related to This Item", style=STYLES['recommendations_title']),
            html.Div([
                create_mini_product_card(i, label)
                for i in similar_indices[:4]
            ], style=STYLES['product_grid']),
        ]))
    
    return html.Div(sections, style=STYLES['recommendations_section'])

def create_mini_product_card(idx, label):
    """Create a smaller product card for recommendations"""
    product_data = generate_product_data(idx, label)
    return html.Div([
        html.Div([
            html.Img(src=get_image(idx), style=STYLES['product_image']),
        ], style=STYLES['product_image_container']),
        html.Div([
            html.Div(f"Product #{idx}", style=STYLES['product_name']),
            html.Div([
                html.Span("★" * int(product_data['rating']), style=STYLES['stars']),
                html.Span(f"({product_data['reviews']})", style=STYLES['review_count']),
            ], style=STYLES['product_rating']),
            html.Div(f"${product_data['price']}", style=STYLES['product_price']),
        ], style=STYLES['product_info']),
    ], style=STYLES['product_card'], id={'type': 'rec-product', 'index': idx})

def create_cart_view(cart, wishlist):
    """Create cart view content"""
    if not cart:
        return html.Div([
            html.H2("Shopping Cart", style={**STYLES['section_title'], 'marginBottom': '32px'}),
            html.Div([
                html.Div("🛒", style=STYLES['empty_icon']),
                html.P("Your cart is empty", style=STYLES['empty_text']),
                html.Button("Continue Shopping", id='cart-continue-shopping', style=STYLES['primary_btn']),
            ], style=STYLES['empty_state']),
        ], style=STYLES['cart_container'])
    
    # Calculate totals
    subtotal = sum(generate_product_data(item['idx'], item['label'])['price'] * item['quantity'] for item in cart)
    shipping = 0 if subtotal >= 100 else 9.99
    total = subtotal + shipping
    
    return html.Div([
        html.H2("Shopping Cart", style={**STYLES['section_title'], 'marginBottom': '32px'}),
        
        # Cart items
        html.Div([
            html.Div([
                html.Img(src=get_image(item['idx']), style=STYLES['cart_item_image']),
                html.Div([
                    html.H3(f"Product #{item['idx']}", style={'fontWeight': '500', 'marginBottom': '4px'}),
                    html.P(f"${generate_product_data(item['idx'], item['label'])['price']}", 
                           style={'color': COLORS['stone_500'], 'marginBottom': '12px'}),
                    html.Div([
                        html.Button("−", id={'type': 'cart-decrease', 'idx': item['idx']}, style=STYLES['quantity_btn']),
                        html.Span(str(item['quantity']), style={'width': '32px', 'textAlign': 'center'}),
                        html.Button("+", id={'type': 'cart-increase', 'idx': item['idx']}, style=STYLES['quantity_btn']),
                    ], style=STYLES['quantity_controls']),
                ], style=STYLES['cart_item_info']),
                html.Button("✕", id={'type': 'cart-remove', 'idx': item['idx']}, 
                           style={'border': 'none', 'background': 'none', 'cursor': 'pointer', 
                                  'color': COLORS['stone_400'], 'fontSize': '20px'}),
            ], style=STYLES['cart_item'])
            for item in cart
        ]),
        
        # Cart summary
        html.Div([
            html.Div([
                html.Span("Subtotal", style={'color': COLORS['stone_600']}),
                html.Span(f"${subtotal:.2f}"),
            ], style=STYLES['summary_row']),
            html.Div([
                html.Span("Shipping", style={'color': COLORS['stone_600']}),
                html.Span("Free" if shipping == 0 else f"${shipping:.2f}"),
            ], style=STYLES['summary_row']),
            html.Div([
                html.Span("Total"),
                html.Span(f"${total:.2f}"),
            ], style=STYLES['summary_total']),
            html.Button("Checkout", id='checkout-btn', 
                       style={**STYLES['primary_btn'], 'width': '100%', 'marginTop': '24px', 'padding': '16px'}),
            html.P("Free shipping on orders over $100", 
                   style={'textAlign': 'center', 'fontSize': '12px', 'color': COLORS['stone_500'], 'marginTop': '16px'}),
        ], style=STYLES['cart_summary']),
    ], style=STYLES['cart_container'])

def create_wishlist_view(cart, wishlist):
    """Create wishlist view content"""
    if not wishlist:
        return html.Div([
            html.H2("Wishlist", style={**STYLES['section_title'], 'marginBottom': '32px'}),
            html.Div([
                html.Div("💝", style=STYLES['empty_icon']),
                html.P("Your wishlist is empty", style=STYLES['empty_text']),
                html.Button("Explore Products", id='wishlist-explore', style=STYLES['primary_btn']),
            ], style=STYLES['empty_state']),
        ], style=STYLES['section'])
    
    return html.Div([
        html.H2("Wishlist", style={**STYLES['section_title'], 'marginBottom': '32px'}),
        html.Div([
            html.Div([
                html.Div([
                    html.Img(src=get_image(idx), style=STYLES['product_image']),
                    html.Button("♥", id={'type': 'wl-remove', 'index': idx},
                               style={'position': 'absolute', 'top': '12px', 'right': '12px',
                                      'width': '32px', 'height': '32px', 'borderRadius': '50%',
                                      'backgroundColor': 'white', 'border': 'none',
                                      'color': COLORS['red_500'], 'cursor': 'pointer'}),
                ], style={**STYLES['product_image_container'], 'position': 'relative'}),
                html.Div([
                    html.Div(f"Product #{idx}", style=STYLES['product_name']),
                    html.Div(f"${generate_product_data(idx, df.iloc[idx]['label'])['price']}", 
                             style={**STYLES['product_price'], 'marginBottom': '12px'}),
                    html.Button("Add to Cart", id={'type': 'wl-add-cart', 'index': idx},
                               style={**STYLES['primary_btn'], 'width': '100%', 'padding': '8px', 'fontSize': '12px'}),
                ], style=STYLES['product_info']),
            ], style=STYLES['product_card'])
            for idx in wishlist if idx < len(df)
        ], style=STYLES['product_grid']),
    ], style=STYLES['section'])

# ============================================================
# APP LAYOUT
# ============================================================

# Custom CSS for hover effects
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Playfair+Display:wght@400;500;600&display=swap');

* { box-sizing: border-box; }

.product-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.product-card:hover .product-image-container img {
    transform: scale(1.05);
}

.product-card:hover .quick-actions {
    opacity: 1 !important;
    transform: translateY(0) !important;
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.2); }
}

.animate-slideUp {
    animation: slideUp 0.4s ease-out forwards;
}

.hide-scrollbar::-webkit-scrollbar {
    display: none;
}

.hide-scrollbar {
    -ms-overflow-style: none;
    scrollbar-width: none;
}
</style>
"""

app.layout = html.Div([
    # Inject custom CSS
    html.Div(dangerously_allow_html=True, children=CUSTOM_CSS) if hasattr(html.Div, 'dangerously_allow_html') else html.Div(),
    dcc.Markdown(CUSTOM_CSS, dangerously_allow_html=True),
    
    # Store for app state
    dcc.Store(id='app-state', data={
        'current_view': 'home',  # home, products, product_detail, cart, wishlist
        'current_category': 'All',
        'current_page': 1,
        'sort_by': 'default',
        'selected_product': None,
        'previous_view': 'home',
    }),
    
    # Store for cart (list of {idx, label, quantity})
    dcc.Store(id='cart-store', data=[]),
    
    # Store for wishlist (list of product indices)
    dcc.Store(id='wishlist-store', data=[]),
    
    # Toast notification state
    dcc.Store(id='toast-state', data={'show': False, 'message': ''}),
    
    # Navigation
    create_navbar(),
    
    # Main content area (updated by callbacks)
    html.Main(id='main-content'),
    
    # Toast notification
    html.Div(id='toast-notification', style={'display': 'none'}),
    
    # Footer (shown conditionally)
    html.Div(id='footer-container'),
    
], style=STYLES['page'])

# ============================================================
# CALLBACKS
# ============================================================

ITEMS_PER_PAGE = 24

@callback(
    Output('main-content', 'children'),
    Output('breadcrumb-area', 'children'),
    Output('results-count', 'children'),
    Output('app-state', 'data'),
    Input('app-state', 'data'),
    Input({'type': 'nav-link', 'category': ALL}, 'n_clicks'),
    Input({'type': 'page-btn', 'page': ALL}, 'n_clicks'),
    Input('search-input', 'value'),
    Input('sort-dropdown', 'value'),
    Input('logo-home', 'n_clicks'),
    Input('breadcrumb-home', 'n_clicks'),
    prevent_initial_call=False,
)
def update_content(state, nav_clicks, page_clicks, search_value, sort_value, logo_click, breadcrumb_click):
    triggered = ctx.triggered_id
    
    # Handle navigation clicks
    if triggered and isinstance(triggered, dict):
        if triggered.get('type') == 'nav-link':
            state['current_category'] = triggered['category']
            state['current_page'] = 1
            state['search_query'] = ''
        elif triggered.get('type') == 'page-btn':
            state['current_page'] = triggered['page']
    
    # Handle logo/home clicks
    if triggered in ['logo-home', 'breadcrumb-home']:
        state['current_category'] = 'All'
        state['current_page'] = 1
        state['search_query'] = ''
    
    # Handle search
    if triggered == 'search-input' and search_value:
        state['search_query'] = search_value.strip()
        state['current_page'] = 1
    
    # Handle sort
    if triggered == 'sort-dropdown' and sort_value:
        state['sort_by'] = sort_value
    
    # Filter products
    category = state.get('current_category', 'All')
    search_query = state.get('search_query', '')
    sort_by = state.get('sort_by', 'default')
    current_page = state.get('current_page', 1)
    
    # Get label indices for category
    label_indices = CATEGORIES.get(category, list(range(10)))
    filtered_df = df[df['label'].isin(label_indices)].copy()
    
    display_category = category
    
    # Apply search filter
    if search_query:
        # Search by label name or product ID
        search_lower = search_query.lower()
        matching_labels = [k for k, v in LABEL_NAMES.items() if search_lower in v.lower()]
        
        # Try to match by ID
        try:
            search_id = int(search_query)
            id_mask = filtered_df.index == search_id
        except ValueError:
            id_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
        
        label_mask = filtered_df['label'].isin(matching_labels)
        filtered_df = filtered_df[label_mask | id_mask]
        display_category = f"Search: '{search_query}'"
    
    # Apply sorting
    if sort_by == 'category_asc':
        filtered_df = filtered_df.sort_values('label')
    elif sort_by == 'category_desc':
        filtered_df = filtered_df.sort_values('label', ascending=False)
    elif sort_by == 'id_asc':
        filtered_df = filtered_df.sort_index()
    elif sort_by == 'id_desc':
        filtered_df = filtered_df.sort_index(ascending=False)
    
    # Pagination
    total_items = len(filtered_df)
    total_pages = max(1, math.ceil(total_items / ITEMS_PER_PAGE))
    current_page = min(current_page, total_pages)
    state['current_page'] = current_page
    
    start_idx = (current_page - 1) * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    page_df = filtered_df.iloc[start_idx:end_idx]
    
    # Build content
    content = []
    
    # Hero (only on homepage)
    if category == 'All' and current_page == 1 and not search_query:
        content.append(create_hero())
        
        # Featured section
        content.append(html.Div([
            html.H2("NEW ARRIVALS", style=STYLES['section_title']),
            html.Div([
                create_product_card(idx, row['label'])
                for idx, row in df.head(8).iterrows()
            ], style=STYLES['product_grid']),
        ], style=STYLES['section']))
        
        content.append(html.Hr(style={'border': 'none', 'borderTop': '1px solid #e0e0e0', 'margin': '0'}))
    
    # Product grid
    if total_items > 0:
        content.append(html.Div([
            html.Div([
                create_product_card(idx, row['label'])
                for idx, row in page_df.iterrows()
            ], style=STYLES['product_grid']),
        ], style=STYLES['section']))
        
        # Pagination
        if total_pages > 1:
            content.append(create_pagination(current_page, total_pages))
    else:
        content.append(html.Div([
            html.P("No products found.", style={'textAlign': 'center', 'padding': '60px', 'color': '#888888'}),
        ]))
    
    # Create breadcrumb
    breadcrumb = create_breadcrumb(display_category)
    
    # Results count
    results_text = f"{total_items} items"
    
    return content, breadcrumb, results_text, state

@callback(
    Output('autocomplete-dropdown', 'children'),
    Output('autocomplete-dropdown', 'style'),
    Input('search-input', 'value'),
    prevent_initial_call=True,
)
def update_autocomplete(search_value):
    if not search_value or len(search_value) < 2:
        return [], {'display': 'none'}
    
    search_lower = search_value.lower()
    suggestions = []
    
    # Category suggestions
    for label_id, label_name in LABEL_NAMES.items():
        if search_lower in label_name.lower():
            count = len(df[df['label'] == label_id])
            suggestions.append({
                'text': f"{label_name} ({count} items)",
                'value': label_name,
            })
    
    if not suggestions:
        return [], {'display': 'none'}
    
    dropdown_items = [
        html.Div(
            s['text'],
            style=STYLES['autocomplete_item'],
            id={'type': 'autocomplete-item', 'value': s['value']},
        )
        for s in suggestions[:5]
    ]
    
    return dropdown_items, {**STYLES['autocomplete_dropdown'], 'display': 'block'}

@callback(
    Output('product-modal', 'children'),
    Output('product-modal', 'style'),
    Input({'type': 'product-card', 'index': ALL}, 'n_clicks'),
    Input({'type': 'similar-item', 'index': ALL}, 'n_clicks'),
    Input('product-modal', 'n_clicks'),
    prevent_initial_call=True,
)
def show_product_modal(card_clicks, similar_clicks, modal_click):
    triggered = ctx.triggered_id
    
    # Close modal if clicking overlay
    if triggered == 'product-modal':
        return [], {'display': 'none'}
    
    # Get clicked product index
    if isinstance(triggered, dict):
        idx = triggered.get('index')
        if idx is not None and any(card_clicks) or any(similar_clicks):
            label = df.iloc[idx]['label']
            
            modal_content = html.Div([
                html.Div([
                    html.Span("×", style=STYLES['modal_close'], id='modal-close-btn'),
                    html.Div([
                        html.Div([
                            html.Img(
                                src=get_image(idx),
                                style={'width': '300px', 'height': '300px', 'objectFit': 'contain', 'backgroundColor': '#fafafa'},
                            ),
                        ], style={'flex': '1'}),
                        html.Div([
                            html.H2(f"Product #{idx}", style={'fontSize': '24px', 'marginBottom': '10px'}),
                            html.P(LABEL_NAMES[label], style={'fontSize': '16px', 'color': '#888', 'textTransform': 'uppercase', 'letterSpacing': '2px'}),
                            html.P(f"Product ID: {idx:05d}", style={'fontSize': '14px', 'color': '#aaa', 'marginTop': '20px'}),
                            html.P(f"Category: {LABEL_NAMES[label]}", style={'fontSize': '14px', 'color': '#666', 'marginTop': '10px'}),
                            html.Hr(style={'margin': '30px 0', 'border': 'none', 'borderTop': '1px solid #e0e0e0'}),
                            html.P("PRODUCT DETAILS", style={'fontSize': '12px', 'fontWeight': '600', 'letterSpacing': '1px', 'marginBottom': '10px'}),
                            html.P("This is a sample product from our collection. Additional details such as price, size, and color would appear here.", 
                                  style={'fontSize': '14px', 'color': '#666', 'lineHeight': '1.6'}),
                        ], style={'flex': '1', 'paddingLeft': '40px'}),
                    ], style={'display': 'flex', 'gap': '20px'}),
                    
                    # Similar items
                    create_similar_items(idx, label),
                ], style={**STYLES['modal_content'], 'position': 'relative'}),
            ], style=STYLES['modal_overlay'], id='product-modal-inner')
            
            return modal_content, {'display': 'block'}
    
    return [], {'display': 'none'}

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == '__main__':
    print("Starting MONO Fashion E-Commerce...")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run(debug=True, port=8050)