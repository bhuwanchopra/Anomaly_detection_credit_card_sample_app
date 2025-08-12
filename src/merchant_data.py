"""
Merchant data and categories for transaction generation
"""

import random

# Merchant categories with typical spending patterns
MERCHANT_CATEGORIES = {
    'grocery': {
        'names': [
            'Fresh Market', 'SuperSave Grocery', 'Green Valley Foods',
            'City Market', 'Organic Plus', 'Corner Store', 'FoodMart',
            'Whole Foods Market', 'Trader Joe\'s', 'Safeway'
        ],
        'typical_amount_range': (5.0, 150.0),
        'peak_hours': [8, 9, 17, 18, 19]
    },
    'gas_station': {
        'names': [
            'Shell Station', 'Exxon Mobil', 'BP Gas', 'Chevron',
            'Speedway', 'Circle K', 'Wawa', '7-Eleven', 'QuikTrip'
        ],
        'typical_amount_range': (20.0, 100.0),
        'peak_hours': [7, 8, 17, 18]
    },
    'restaurant': {
        'names': [
            'The Grill House', 'Pasta Palace', 'Burger Joint', 'Sushi Bar',
            'Pizza Corner', 'Cafe Bistro', 'Steakhouse Prime', 'Taco Bell',
            'McDonald\'s', 'Subway', 'Italian Garden', 'Thai Kitchen'
        ],
        'typical_amount_range': (8.0, 120.0),
        'peak_hours': [12, 13, 18, 19, 20]
    },
    'retail': {
        'names': [
            'Fashion Forward', 'Electronics Plus', 'Home Depot', 'Target',
            'Walmart', 'Best Buy', 'Macy\'s', 'Nike Store', 'Apple Store',
            'Amazon Store', 'Costco', 'TJ Maxx'
        ],
        'typical_amount_range': (15.0, 500.0),
        'peak_hours': [14, 15, 16, 19, 20]
    },
    'online': {
        'names': [
            'Amazon.com', 'eBay', 'PayPal', 'Netflix', 'Spotify',
            'Apple iTunes', 'Google Play', 'Steam', 'Uber', 'Lyft'
        ],
        'typical_amount_range': (5.0, 200.0),
        'peak_hours': list(range(24))  # Online transactions can happen anytime
    },
    'entertainment': {
        'names': [
            'Movie Theater', 'Concert Hall', 'Sports Arena', 'Bowling Alley',
            'Game Zone', 'Mini Golf', 'Amusement Park', 'Casino',
            'Dave & Busters', 'AMC Theaters'
        ],
        'typical_amount_range': (10.0, 150.0),
        'peak_hours': [18, 19, 20, 21, 22]
    },
    'healthcare': {
        'names': [
            'Family Clinic', 'Dental Care', 'Pharmacy Plus', 'CVS Pharmacy',
            'Walgreens', 'Hospital Center', 'Urgent Care', 'Vision Center'
        ],
        'typical_amount_range': (20.0, 300.0),
        'peak_hours': [9, 10, 11, 14, 15, 16]
    },
    'travel': {
        'names': [
            'Delta Airlines', 'United Airlines', 'Hilton Hotel', 'Marriott',
            'Hertz Car Rental', 'Enterprise', 'Airbnb', 'Expedia',
            'Hotels.com', 'Southwest Airlines'
        ],
        'typical_amount_range': (50.0, 1000.0),
        'peak_hours': [9, 10, 11, 13, 14, 15]
    }
}

# US Cities and States for realistic location data
US_LOCATIONS = [
    ('New York', 'NY'), ('Los Angeles', 'CA'), ('Chicago', 'IL'),
    ('Houston', 'TX'), ('Phoenix', 'AZ'), ('Philadelphia', 'PA'),
    ('San Antonio', 'TX'), ('San Diego', 'CA'), ('Dallas', 'TX'),
    ('San Jose', 'CA'), ('Austin', 'TX'), ('Jacksonville', 'FL'),
    ('Fort Worth', 'TX'), ('Columbus', 'OH'), ('Charlotte', 'NC'),
    ('San Francisco', 'CA'), ('Indianapolis', 'IN'), ('Seattle', 'WA'),
    ('Denver', 'CO'), ('Washington', 'DC'), ('Boston', 'MA'),
    ('El Paso', 'TX'), ('Detroit', 'MI'), ('Nashville', 'TN'),
    ('Portland', 'OR'), ('Memphis', 'TN'), ('Oklahoma City', 'OK'),
    ('Las Vegas', 'NV'), ('Louisville', 'KY'), ('Baltimore', 'MD'),
    ('Milwaukee', 'WI'), ('Albuquerque', 'NM'), ('Tucson', 'AZ'),
    ('Fresno', 'CA'), ('Sacramento', 'CA'), ('Mesa', 'AZ'),
    ('Kansas City', 'MO'), ('Atlanta', 'GA'), ('Long Beach', 'CA'),
    ('Colorado Springs', 'CO'), ('Raleigh', 'NC'), ('Miami', 'FL'),
    ('Virginia Beach', 'VA'), ('Omaha', 'NE'), ('Oakland', 'CA'),
    ('Minneapolis', 'MN'), ('Tulsa', 'OK'), ('Arlington', 'TX'),
    ('Tampa', 'FL'), ('New Orleans', 'LA')
]


def get_random_merchant(category=None):
    """Get a random merchant from a specific category or any category."""
    if category:
        if category not in MERCHANT_CATEGORIES:
            raise ValueError(f"Unknown category: {category}")
        cat_data = MERCHANT_CATEGORIES[category]
        return {
            'name': random.choice(cat_data['names']),
            'category': category,
            'amount_range': cat_data['typical_amount_range'],
            'peak_hours': cat_data['peak_hours']
        }
    else:
        category = random.choice(list(MERCHANT_CATEGORIES.keys()))
        return get_random_merchant(category)


def get_random_location():
    """Get a random US city and state."""
    city, state = random.choice(US_LOCATIONS)
    return city, state, 'USA'


def get_category_spending_pattern(category):
    """Get spending pattern for a specific category."""
    return MERCHANT_CATEGORIES.get(category, {
        'typical_amount_range': (10.0, 100.0),
        'peak_hours': [12, 13, 18, 19]
    })
