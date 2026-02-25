"""
Fashion E-Commerce Website using Dash
A basic, plain website to display clothing items from pixel data CSV
"""

import dash
from dash import html, dcc, callback, Input, Output, State, ctx, ALL
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import math
import cv2
import pickle
import os

IMAGE_CACHE_FILE = 'image_cache.pkl'

# ============================================================
# DATA LOADING AND IMAGE PROCESSING
# ============================================================

LABEL_NAMES = {
    0: 'T-shirt/Top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'
}

# Initialize EDSR super-resolution model
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("EDSR_x4.pb")
sr.setModel("edsr", 4)
print("EDSR super-resolution model loaded.")

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
    """Convert 784 pixel values to base64 encoded image using EDSR super-resolution"""
    img_array = np.array(pixel_array).reshape(28, 28).astype(np.uint8)
    # Invert: lower values = lighter in data, but we want dark items on light bg
    img_array = 255 - img_array
    
    # Convert grayscale to BGR for EDSR (requires 3 channels)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    # Upscale using EDSR super-resolution (4x: 28x28 -> 112x112)
    upscaled = sr.upsample(img_bgr)
    
    # Convert back to grayscale
    upscaled_gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    
    # Convert to PIL Image and save as PNG
    img = Image.fromarray(upscaled_gray, mode='L')
    
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Load data at startup
df = load_data()
pixel_cols = [f'pixel_{i}' for i in range(784)]

def load_image_cache():
    """Load image cache from disk if it exists and matches current data size"""
    if os.path.exists(IMAGE_CACHE_FILE):
        try:
            with open(IMAGE_CACHE_FILE, 'rb') as f:
                cached_data = pickle.load(f)
            # Verify cache matches current dataset size
            if len(cached_data) == len(df):
                print(f"Loaded {len(cached_data)} images from cache.")
                return cached_data
            else:
                print("Cache size mismatch, regenerating...")
        except Exception as e:
            print(f"Cache load failed: {e}, regenerating...")
    return None

def save_image_cache(cache):
    """Save image cache to disk"""
    try:
        with open(IMAGE_CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
        print(f"Saved {len(cache)} images to cache.")
    except Exception as e:
        print(f"Failed to save cache: {e}")

# Try to load from disk cache first
image_cache = load_image_cache()

if image_cache is None:
    # Generate images with EDSR (slow, first time only)
    print("Converting images with EDSR super-resolution (first run, please wait)...")
    image_cache = {}
    total = len(df)
    for idx in range(total):
        if idx % 100 == 0:
            print(f"Processing image {idx}/{total}...")
        image_cache[idx] = pixels_to_base64(df.iloc[idx][pixel_cols].values)
    save_image_cache(image_cache)
    print("Image conversion complete.")
else:
    print("Using cached images (fast load).")

# ============================================================
# DASH APP INITIALIZATION
# ============================================================

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "MONO Fashion"

# ============================================================
# STYLES (Plain, minimal CSS)
# ============================================================

STYLES = {
    'page': {
        'fontFamily': 'Arial, sans-serif',
        'backgroundColor': '#ffffff',
        'minHeight': '100vh',
        'color': '#333333',
    },
    'navbar': {
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'padding': '15px 40px',
        'borderBottom': '1px solid #e0e0e0',
        'backgroundColor': '#ffffff',
        'position': 'sticky',
        'top': 0,
        'zIndex': 1000,
    },
    'logo': {
        'fontSize': '24px',
        'fontWeight': 'bold',
        'letterSpacing': '3px',
        'color': '#000000',
        'textDecoration': 'none',
        'cursor': 'pointer',
    },
    'nav_links': {
        'display': 'flex',
        'gap': '30px',
        'listStyle': 'none',
        'margin': 0,
        'padding': 0,
    },
    'nav_link': {
        'textDecoration': 'none',
        'color': '#333333',
        'fontSize': '14px',
        'textTransform': 'uppercase',
        'letterSpacing': '1px',
        'cursor': 'pointer',
        'padding': '5px 0',
        'borderBottom': '2px solid transparent',
    },
    'nav_link_active': {
        'borderBottom': '2px solid #000000',
    },
    'search_container': {
        'position': 'relative',
        'width': '250px',
    },
    'search_input': {
        'width': '100%',
        'padding': '10px 15px',
        'border': '1px solid #e0e0e0',
        'borderRadius': '0',
        'fontSize': '14px',
        'outline': 'none',
    },
    'hero': {
        'backgroundColor': '#f5f5f5',
        'padding': '60px 40px',
        'textAlign': 'center',
        'borderBottom': '1px solid #e0e0e0',
    },
    'hero_title': {
        'fontSize': '36px',
        'fontWeight': '300',
        'letterSpacing': '5px',
        'marginBottom': '15px',
        'color': '#000000',
    },
    'hero_subtitle': {
        'fontSize': '14px',
        'color': '#666666',
        'letterSpacing': '2px',
    },
    'section': {
        'padding': '40px',
    },
    'section_title': {
        'fontSize': '18px',
        'fontWeight': '400',
        'letterSpacing': '2px',
        'textTransform': 'uppercase',
        'marginBottom': '30px',
        'color': '#000000',
        'borderBottom': '1px solid #e0e0e0',
        'paddingBottom': '10px',
    },
    'product_grid': {
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fill, minmax(200px, 1fr))',
        'gap': '30px',
    },
    'product_card': {
        'border': '1px solid #e0e0e0',
        'padding': '15px',
        'textAlign': 'center',
        'cursor': 'pointer',
        'transition': 'box-shadow 0.2s',
        'backgroundColor': '#ffffff',
    },
    'product_image': {
        'width': '100%',
        'height': '180px',
        'objectFit': 'contain',
        'backgroundColor': '#fafafa',
        'marginBottom': '15px',
    },
    'product_name': {
        'fontSize': '14px',
        'fontWeight': '400',
        'color': '#333333',
        'marginBottom': '5px',
    },
    'product_category': {
        'fontSize': '12px',
        'color': '#888888',
        'textTransform': 'uppercase',
        'letterSpacing': '1px',
    },
    'product_id': {
        'fontSize': '11px',
        'color': '#aaaaaa',
        'marginTop': '5px',
    },
    'breadcrumb': {
        'padding': '15px 40px',
        'fontSize': '12px',
        'color': '#888888',
        'borderBottom': '1px solid #e0e0e0',
    },
    'breadcrumb_link': {
        'color': '#888888',
        'textDecoration': 'none',
        'cursor': 'pointer',
    },
    'pagination': {
        'display': 'flex',
        'justifyContent': 'center',
        'gap': '10px',
        'padding': '40px',
    },
    'page_btn': {
        'padding': '10px 15px',
        'border': '1px solid #e0e0e0',
        'backgroundColor': '#ffffff',
        'cursor': 'pointer',
        'fontSize': '14px',
    },
    'page_btn_active': {
        'backgroundColor': '#000000',
        'color': '#ffffff',
        'border': '1px solid #000000',
    },
    'footer': {
        'backgroundColor': '#f5f5f5',
        'padding': '40px',
        'borderTop': '1px solid #e0e0e0',
        'marginTop': '40px',
    },
    'footer_grid': {
        'display': 'grid',
        'gridTemplateColumns': 'repeat(4, 1fr)',
        'gap': '40px',
        'maxWidth': '1200px',
        'margin': '0 auto',
    },
    'footer_title': {
        'fontSize': '12px',
        'fontWeight': '600',
        'textTransform': 'uppercase',
        'letterSpacing': '1px',
        'marginBottom': '15px',
        'color': '#333333',
    },
    'footer_link': {
        'fontSize': '13px',
        'color': '#666666',
        'textDecoration': 'none',
        'display': 'block',
        'marginBottom': '8px',
        'cursor': 'pointer',
    },
    'footer_bottom': {
        'textAlign': 'center',
        'paddingTop': '30px',
        'marginTop': '30px',
        'borderTop': '1px solid #e0e0e0',
        'fontSize': '12px',
        'color': '#888888',
    },
    'similar_section': {
        'padding': '20px 40px',
        'backgroundColor': '#fafafa',
        'borderTop': '1px solid #e0e0e0',
    },
    'similar_title': {
        'fontSize': '14px',
        'fontWeight': '400',
        'letterSpacing': '1px',
        'textTransform': 'uppercase',
        'marginBottom': '20px',
        'color': '#666666',
    },
    'similar_carousel': {
        'display': 'flex',
        'gap': '20px',
        'overflowX': 'auto',
        'paddingBottom': '15px',
    },
    'similar_item': {
        'minWidth': '150px',
        'textAlign': 'center',
        'cursor': 'pointer',
    },
    'similar_image': {
        'width': '120px',
        'height': '120px',
        'objectFit': 'contain',
        'backgroundColor': '#ffffff',
        'border': '1px solid #e0e0e0',
    },
    'filter_bar': {
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'padding': '15px 40px',
        'backgroundColor': '#fafafa',
        'borderBottom': '1px solid #e0e0e0',
    },
    'filter_dropdown': {
        'padding': '8px 15px',
        'border': '1px solid #e0e0e0',
        'backgroundColor': '#ffffff',
        'fontSize': '13px',
        'cursor': 'pointer',
    },
    'results_count': {
        'fontSize': '13px',
        'color': '#666666',
    },
    'autocomplete_dropdown': {
        'position': 'absolute',
        'top': '100%',
        'left': 0,
        'right': 0,
        'backgroundColor': '#ffffff',
        'border': '1px solid #e0e0e0',
        'borderTop': 'none',
        'zIndex': 1001,
        'maxHeight': '200px',
        'overflowY': 'auto',
    },
    'autocomplete_item': {
        'padding': '10px 15px',
        'cursor': 'pointer',
        'fontSize': '13px',
        'borderBottom': '1px solid #f0f0f0',
    },
    'modal_overlay': {
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'right': 0,
        'bottom': 0,
        'backgroundColor': 'rgba(0,0,0,0.5)',
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'center',
        'zIndex': 2000,
    },
    'modal_content': {
        'backgroundColor': '#ffffff',
        'padding': '40px',
        'maxWidth': '800px',
        'width': '90%',
        'maxHeight': '90vh',
        'overflowY': 'auto',
    },
    'modal_close': {
        'position': 'absolute',
        'top': '15px',
        'right': '20px',
        'fontSize': '24px',
        'cursor': 'pointer',
        'color': '#666666',
    },
}

# ============================================================
# COMPONENTS
# ============================================================

def create_navbar():
    """Create navigation bar"""
    return html.Div([
        html.Div("MONO", style=STYLES['logo'], id='logo-home'),
        html.Ul([
            html.Li(html.A("All", id={'type': 'nav-link', 'category': 'All'}), 
                   style={'listStyle': 'none'}),
            html.Li(html.A("Tops", id={'type': 'nav-link', 'category': 'Tops'}), 
                   style={'listStyle': 'none'}),
            html.Li(html.A("Bottoms", id={'type': 'nav-link', 'category': 'Bottoms'}), 
                   style={'listStyle': 'none'}),
            html.Li(html.A("Dresses", id={'type': 'nav-link', 'category': 'Dresses'}), 
                   style={'listStyle': 'none'}),
            html.Li(html.A("Footwear", id={'type': 'nav-link', 'category': 'Footwear'}), 
                   style={'listStyle': 'none'}),
            html.Li(html.A("Accessories", id={'type': 'nav-link', 'category': 'Accessories'}), 
                   style={'listStyle': 'none'}),
        ], style=STYLES['nav_links']),
        html.Div([
            dcc.Input(
                id='search-input',
                type='text',
                placeholder='Search products...',
                style=STYLES['search_input'],
                debounce=True,
            ),
            html.Div(id='autocomplete-dropdown', style={'display': 'none'}),
        ], style=STYLES['search_container']),
    ], style=STYLES['navbar'])

def create_hero():
    """Create hero banner"""
    return html.Div([
        html.H1("NEW COLLECTION", style=STYLES['hero_title']),
        html.P("DISCOVER OUR LATEST ARRIVALS", style=STYLES['hero_subtitle']),
    ], style=STYLES['hero'])

def create_product_card(idx, label):
    """Create a single product card"""
    return html.Div([
        html.Img(
            src=image_cache[idx],
            style=STYLES['product_image'],
        ),
        html.Div(f"Product #{idx}", style=STYLES['product_name']),
        html.Div(LABEL_NAMES[label], style=STYLES['product_category']),
        html.Div(f"ID: {idx:05d}", style=STYLES['product_id']),
    ], style=STYLES['product_card'], id={'type': 'product-card', 'index': idx})

def create_footer():
    """Create footer"""
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
        html.Div("© 2024 MONO Fashion. All rights reserved.", style=STYLES['footer_bottom']),
    ], style=STYLES['footer'])

def create_breadcrumb(category='All'):
    """Create breadcrumb navigation"""
    return html.Div([
        html.Span("Home", style=STYLES['breadcrumb_link'], id='breadcrumb-home'),
        html.Span(" / ", style={'margin': '0 10px'}),
        html.Span(category, style={'color': '#333333'}),
    ], style=STYLES['breadcrumb'])

def create_filter_bar(total_items, category='All'):
    """Create filter and sort bar"""
    return html.Div([
        html.Span(f"{total_items} items", style=STYLES['results_count']),
        html.Div([
            html.Select([
                html.Option("Sort by: Default", value='default'),
                html.Option("Sort by: Category A-Z", value='category_asc'),
                html.Option("Sort by: Category Z-A", value='category_desc'),
                html.Option("Sort by: ID (Low-High)", value='id_asc'),
                html.Option("Sort by: ID (High-Low)", value='id_desc'),
            ], id='sort-dropdown', value='default', style=STYLES['filter_dropdown']),
        ]),
    ], style=STYLES['filter_bar'])

def create_similar_items(current_idx, label):
    """Create carousel of similar items from same category"""
    similar_indices = df[df['label'] == label].index.tolist()
    similar_indices = [i for i in similar_indices if i != current_idx][:10]
    
    return html.Div([
        html.Div("SIMILAR ITEMS", style=STYLES['similar_title']),
        html.Div([
            html.Div([
                html.Img(src=image_cache[idx], style=STYLES['similar_image']),
                html.Div(f"#{idx}", style={'fontSize': '11px', 'color': '#888', 'marginTop': '5px'}),
            ], style=STYLES['similar_item'], id={'type': 'similar-item', 'index': idx})
            for idx in similar_indices
        ], style=STYLES['similar_carousel']),
    ], style=STYLES['similar_section'])

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
    
    # Page numbers (show max 5 pages around current)
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

# ============================================================
# APP LAYOUT
# ============================================================

app.layout = html.Div([
    # Store for app state
    dcc.Store(id='app-state', data={
        'current_category': 'All',
        'current_page': 1,
        'search_query': '',
        'sort_by': 'default',
        'selected_product': None,
    }),
    
    # Navigation
    create_navbar(),
    
    # Breadcrumb area
    html.Div(create_breadcrumb('All'), id='breadcrumb-area'),
    
    # Filter bar (always present)
    html.Div([
        html.Span(id='results-count', style=STYLES['results_count']),
        html.Div([
            dcc.Dropdown(
                id='sort-dropdown',
                options=[
                    {'label': 'Sort by: Default', 'value': 'default'},
                    {'label': 'Sort by: Category A-Z', 'value': 'category_asc'},
                    {'label': 'Sort by: Category Z-A', 'value': 'category_desc'},
                    {'label': 'Sort by: ID (Low-High)', 'value': 'id_asc'},
                    {'label': 'Sort by: ID (High-Low)', 'value': 'id_desc'},
                ],
                value='default',
                clearable=False,
                style={'width': '200px', 'fontSize': '13px'},
            ),
        ]),
    ], id='filter-bar', style=STYLES['filter_bar']),
    
    # Main content area
    html.Div(id='main-content'),
    
    # Product detail modal
    html.Div(id='product-modal', style={'display': 'none'}),
    
    # Footer
    create_footer(),
    
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
                                src=image_cache[idx],
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