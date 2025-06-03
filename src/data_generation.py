#src/data_generation.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import random
from src.utils import setup_logging

logger = setup_logging()

def generate_food_dataset(n_samples=50000):
    """
    Generate a synthetic dataset of Ethiopian and European foods with nutritional data, including glucose content, glycemic index, and glycemic load.
    Args:
        n_samples (int): Number of dataset entries.
    Returns:
        pd.DataFrame: Dataset with food names, categories, and nutritional info.
    """
    try:
        # Ethiopian foods dataset (63 foods, including 20 breads)
        ethiopian_foods = [
            {"name": "Injera", "category": "Bread", "carb_range": (50, 60), "calorie_range": (200, 250), "protein_range": (4, 7), "fat_range": (1, 3), "gi_range": (50, 57)},
            {"name": "Doro Wat", "category": "Stew", "carb_range": (5, 15), "calorie_range": (150, 200), "protein_range": (10, 15), "fat_range": (8, 12), "gi_range": (40, 50)},
            {"name": "Tibs", "category": "Meat Dish", "carb_range": (0, 5), "calorie_range": (180, 220), "protein_range": (20, 25), "fat_range": (10, 15), "gi_range": (0, 10)},
            {"name": "Shiro", "category": "Stew", "carb_range": (20, 30), "calorie_range": (120, 160), "protein_range": (8, 12), "fat_range": (5, 8), "gi_range": (45, 55)},
            {"name": "Kitfo", "category": "Meat Dish", "carb_range": (0, 5), "calorie_range": (200, 250), "protein_range": (18, 22), "fat_range": (15, 20), "gi_range": (0, 10)},
            {"name": "Misir Wat", "category": "Stew", "carb_range": (25, 35), "calorie_range": (100, 140), "protein_range": (6, 10), "fat_range": (3, 6), "gi_range": (50, 60)},
            {"name": "Gomen", "category": "Vegetable", "carb_range": (5, 10), "calorie_range": (50, 80), "protein_range": (2, 4), "fat_range": (1, 3), "gi_range": (30, 40)},
            {"name": "Ayib", "category": "Cheese", "carb_range": (0, 3), "calorie_range": (100, 130), "protein_range": (8, 12), "fat_range": (7, 10), "gi_range": (0, 10)},
            {"name": "Teff Porridge", "category": "Porridge", "carb_range": (40, 50), "calorie_range": (150, 180), "protein_range": (5, 8), "fat_range": (2, 4), "gi_range": (50, 60)},
            {"name": "Fitfit", "category": "Bread", "carb_range": (45, 55), "calorie_range": (180, 220), "protein_range": (4, 7), "fat_range": (2, 5), "gi_range": (55, 65)},
            {"name": "Atakilt Wat", "category": "Vegetable", "carb_range": (15, 25), "calorie_range": (80, 120), "protein_range": (2, 5), "fat_range": (3, 6), "gi_range": (35, 45)},
            {"name": "Segwat", "category": "Meat Dish", "carb_range": (0, 5), "calorie_range": (170, 210), "protein_range": (18, 23), "fat_range": (9, 14), "gi_range": (0, 10)},
            {"name": "Fossolia", "category": "Vegetable", "carb_range": (10, 20), "calorie_range": (70, 100), "protein_range": (2, 4), "fat_range": (2, 5), "gi_range": (30, 40)},
            {"name": "Chechebsa", "category": "Bread", "carb_range": (40, 50), "calorie_range": (200, 240), "protein_range": (5, 8), "fat_range": (6, 9), "gi_range": (50, 60)},
            {"name": "Awaze Tibs", "category": "Meat Dish", "carb_range": (2, 8), "calorie_range": (190, 230), "protein_range": (20, 25), "fat_range": (12, 17), "gi_range": (10, 20)},
            {"name": "Dulet", "category": "Meat Dish", "carb_range": (0, 5), "calorie_range": (220, 260), "protein_range": (15, 20), "fat_range": (15, 20), "gi_range": (0, 10)},
            {"name": "Alicha Wat", "category": "Stew", "carb_range": (10, 20), "calorie_range": (90, 130), "protein_range": (3, 6), "fat_range": (3, 6), "gi_range": (40, 50)},
            {"name": "Minchet Abish", "category": "Meat Dish", "carb_range": (5, 10), "calorie_range": (180, 220), "protein_range": (15, 20), "fat_range": (10, 15), "gi_range": (10, 20)},
            {"name": "Kikil", "category": "Soup", "carb_range": (10, 20), "calorie_range": (80, 120), "protein_range": (5, 8), "fat_range": (2, 5), "gi_range": (30, 40)},
            {"name": "Timatim Fitfit", "category": "Salad", "carb_range": (30, 40), "calorie_range": (120, 160), "protein_range": (3, 6), "fat_range": (2, 5), "gi_range": (45, 55)},
            {"name": "Buticha", "category": "Side Dish", "carb_range": (15, 25), "calorie_range": (100, 140), "protein_range": (5, 8), "fat_range": (3, 6), "gi_range": (40, 50)},
            {"name": "Azifa", "category": "Salad", "carb_range": (20, 30), "calorie_range": (90, 130), "protein_range": (5, 8), "fat_range": (2, 5), "gi_range": (45, 55)},
            {"name": "Genfo", "category": "Porridge", "carb_range": (35, 45), "calorie_range": (140, 180), "protein_range": (4, 7), "fat_range": (2, 4), "gi_range": (50, 60)},
            {"name": "Fatira", "category": "Bread", "carb_range": (40, 50), "calorie_range": (200, 240), "protein_range": (5, 8), "fat_range": (6, 10), "gi_range": (55, 65)},
            {"name": "Key Wat", "category": "Stew", "carb_range": (5, 15), "calorie_range": (160, 200), "protein_range": (12, 17), "fat_range": (9, 13), "gi_range": (40, 50)},
            {"name": "Dinich Wat", "category": "Stew", "carb_range": (20, 30), "calorie_range": (90, 130), "protein_range": (2, 5), "fat_range": (3, 6), "gi_range": (40, 50)},
            {"name": "Suf Fitfit", "category": "Side Dish", "carb_range": (30, 40), "calorie_range": (140, 180), "protein_range": (4, 7), "fat_range": (3, 6), "gi_range": (50, 60)},
            {"name": "Yetsom Beyaynetu", "category": "Vegetable", "carb_range": (25, 35), "calorie_range": (120, 160), "protein_range": (5, 8), "fat_range": (3, 6), "gi_range": (40, 50)},
            {"name": "Shorba", "category": "Soup", "carb_range": (10, 20), "calorie_range": (70, 100), "protein_range": (3, 6), "fat_range": (1, 3), "gi_range": (30, 40)},
            {"name": "Anbabero", "category": "Bread", "carb_range": (45, 55), "calorie_range": (190, 230), "protein_range": (4, 7), "fat_range": (2, 5), "gi_range": (50, 60)},
            {"name": "Bula", "category": "Porridge", "carb_range": (35, 45), "calorie_range": (130, 170), "protein_range": (3, 6), "fat_range": (1, 3), "gi_range": (50, 60)},
            {"name": "Gored Gored", "category": "Meat Dish", "carb_range": (0, 5), "calorie_range": (210, 250), "protein_range": (18, 23), "fat_range": (14, 18), "gi_range": (0, 10)},
            {"name": "Sils", "category": "Stew", "carb_range": (10, 20), "calorie_range": (100, 140), "protein_range": (3, 6), "fat_range": (3, 6), "gi_range": (40, 50)},
            {"name": "Tegabino", "category": "Stew", "carb_range": (20, 30), "calorie_range": (130, 170), "protein_range": (8, 12), "fat_range": (5, 8), "gi_range": (45, 55)},
            {"name": "Beyaynetu", "category": "Mixed Dish", "carb_range": (30, 40), "calorie_range": (150, 200), "protein_range": (8, 12), "fat_range": (5, 8), "gi_range": (40, 50)},
            {"name": "Duba Wat", "category": "Vegetable", "carb_range": (15, 25), "calorie_range": (80, 120), "protein_range": (2, 5), "fat_range": (2, 5), "gi_range": (35, 45)},
            {"name": "Enqulal Firfir", "category": "Egg Dish", "carb_range": (5, 10), "calorie_range": (120, 160), "protein_range": (6, 9), "fat_range": (7, 10), "gi_range": (20, 30)},
            {"name": "Defo Dabo", "category": "Bread", "carb_range": (45, 55), "calorie_range": (200, 240), "protein_range": (5, 8), "fat_range": (3, 6), "gi_range": (50, 60)},
            {"name": "Tire Siga", "category": "Meat Dish", "carb_range": (0, 5), "calorie_range": (200, 240), "protein_range": (18, 23), "fat_range": (13, 17), "gi_range": (0, 10)},
            {"name": "Shiro Fitfit", "category": "Side Dish", "carb_range": (35, 45), "calorie_range": (150, 190), "protein_range": (6, 10), "fat_range": (4, 7), "gi_range": (50, 60)},
            {"name": "Kolo", "category": "Snack", "carb_range": (30, 40), "calorie_range": (150, 190), "protein_range": (4, 7), "fat_range": (5, 8), "gi_range": (50, 60)},
            {"name": "Timatim Salad", "category": "Salad", "carb_range": (5, 10), "calorie_range": (40, 70), "protein_range": (1, 3), "fat_range": (1, 3), "gi_range": (20, 30)},
            {"name": "Awaze", "category": "Condiment", "carb_range": (5, 10), "calorie_range": (50, 80), "protein_range": (1, 3), "fat_range": (3, 6), "gi_range": (20, 30)},
            {"name": "Mesir Alicha", "category": "Stew", "carb_range": (25, 35), "calorie_range": (100, 140), "protein_range": (6, 10), "fat_range": (3, 6), "gi_range": (50, 60)},
            {"name": "Ambasha", "category": "Bread", "carb_range": (40, 50), "calorie_range": (190, 230), "protein_range": (4, 7), "fat_range": (3, 6), "gi_range": (50, 60)},
            {"name": "Qanta", "category": "Meat Dish", "carb_range": (0, 3), "calorie_range": (150, 190), "protein_range": (15, 20), "fat_range": (8, 12), "gi_range": (0, 10)},
            {"name": "Gomen Be Siga", "category": "Vegetable", "carb_range": (5, 15), "calorie_range": (100, 140), "protein_range": (5, 8), "fat_range": (5, 8), "gi_range": (30, 40)},
            {"name": "Injera Firfir", "category": "Bread", "carb_range": (45, 55), "calorie_range": (180, 220), "protein_range": (4, 7), "fat_range": (3, 6), "gi_range": (50, 60)},
            {"name": "Telba", "category": "Porridge", "carb_range": (30, 40), "calorie_range": (120, 160), "protein_range": (4, 7), "fat_range": (3, 6), "gi_range": (45, 55)},
            {"name": "Mitmita", "category": "Condiment", "carb_range": (2, 5), "calorie_range": (20, 50), "protein_range": (1, 2), "fat_range": (1, 3), "gi_range": (10, 20)},
            {"name": "Kita", "category": "Bread", "carb_range": (40, 50), "calorie_range": (180, 220), "protein_range": (4, 7), "fat_range": (2, 5), "gi_range": (50, 60)},
            {"name": "Dabo Kolo", "category": "Bread", "carb_range": (35, 45), "calorie_range": (160, 200), "protein_range": (4, 6), "fat_range": (4, 7), "gi_range": (50, 60)},
            {"name": "Himbasha", "category": "Bread", "carb_range": (40, 50), "calorie_range": (190, 230), "protein_range": (4, 7), "fat_range": (3, 6), "gi_range": (50, 60)},
            {"name": "Mulmul", "category": "Bread", "carb_range": (45, 55), "calorie_range": (200, 240), "protein_range": (5, 8), "fat_range": (3, 6), "gi_range": (50, 60)},
            {"name": "Teff Dabo", "category": "Bread", "carb_range": (45, 55), "calorie_range": (190, 230), "protein_range": (5, 8), "fat_range": (2, 5), "gi_range": (50, 57)},
            {"name": "Barley Injera", "category": "Bread", "carb_range": (48, 58), "calorie_range": (190, 230), "protein_range": (4, 7), "fat_range": (1, 3), "gi_range": (55, 62)},
            {"name": "Sorghum Injera", "category": "Bread", "carb_range": (50, 60), "calorie_range": (200, 240), "protein_range": (4, 7), "fat_range": (1, 3), "gi_range": (55, 65)},
            {"name": "Chornake", "category": "Bread", "carb_range": (40, 50), "calorie_range": (180, 220), "protein_range": (4, 7), "fat_range": (2, 5), "gi_range": (50, 60)},
            {"name": "Difo Dabo", "category": "Bread", "carb_range": (45, 55), "calorie_range": (200, 240), "protein_range": (5, 8), "fat_range": (3, 6), "gi_range": (50, 60)},
            {"name": "Enjera Alicha", "category": "Bread", "carb_range": (45, 55), "calorie_range": (180, 220), "protein_range": (4, 7), "fat_range": (2, 5), "gi_range": (50, 57)},
            {"name": "Qurt", "category": "Bread", "carb_range": (40, 50), "calorie_range": (170, 210), "protein_range": (4, 6), "fat_range": (2, 5), "gi_range": (50, 60)},
            {"name": "Shamita", "category": "Bread", "carb_range": (35, 45), "calorie_range": (160, 200), "protein_range": (4, 6), "fat_range": (3, 6), "gi_range": (50, 60)},
            {"name": "Teff Kita", "category": "Bread", "carb_range": (40, 50), "calorie_range": (180, 220), "protein_range": (4, 7), "fat_range": (2, 5), "gi_range": (50, 57)}
        ]

        # European foods dataset (50 foods, including ~20 bakery foods)
        european_foods = [
            {"name": "Pasta", "category": "Pasta", "carb_range": (65, 75), "calorie_range": (300, 350), "protein_range": (10, 14), "fat_range": (1, 3), "gi_range": (40, 50)},
            {"name": "Croissant", "category": "Pastry", "carb_range": (40, 50), "calorie_range": (350, 400), "protein_range": (6, 9), "fat_range": (20, 25), "gi_range": (65, 75)},
            {"name": "Baguette", "category": "Bread", "carb_range": (50, 60), "calorie_range": (250, 300), "protein_range": (8, 12), "fat_range": (1, 3), "gi_range": (70, 80)},
            {"name": "Pizza", "category": "Main Dish", "carb_range": (30, 40), "calorie_range": (250, 300), "protein_range": (10, 15), "fat_range": (10, 15), "gi_range": (45, 55)},
            {"name": "Roast Beef", "category": "Meat Dish", "carb_range": (0, 5), "calorie_range": (200, 250), "protein_range": (25, 30), "fat_range": (10, 15), "gi_range": (0, 10)},
            {"name": "Mashed Potatoes", "category": "Side Dish", "carb_range": (15, 25), "calorie_range": (100, 140), "protein_range": (2, 4), "fat_range": (3, 6), "gi_range": (80, 90)},
            {"name": "Paella", "category": "Main Dish", "carb_range": (20, 30), "calorie_range": (200, 250), "protein_range": (12, 18), "fat_range": (8, 12), "gi_range": (50, 60)},
            {"name": "Tiramisu", "category": "Dessert", "carb_range": (30, 40), "calorie_range": (300, 350), "protein_range": (5, 8), "fat_range": (15, 20), "gi_range": (50, 60)},
            {"name": "Schnitzel", "category": "Meat Dish", "carb_range": (10, 20), "calorie_range": (250, 300), "protein_range": (20, 25), "fat_range": (12, 18), "gi_range": (30, 40)},
            {"name": "Risotto", "category": "Main Dish", "carb_range": (25, 35), "calorie_range": (200, 250), "protein_range": (8, 12), "fat_range": (6, 10), "gi_range": (60, 70)},
            {"name": "Cheeseburger", "category": "Fast Food", "carb_range": (30, 40), "calorie_range": (300, 350), "protein_range": (15, 20), "fat_range": (12, 18), "gi_range": (50, 60)},
            {"name": "burger", "category": "Fast Food", "carb_range": (30, 40), "calorie_range": (300, 350), "protein_range": (15, 20), "fat_range": (12, 18), "gi_range": (50, 60)},
            {"name": "French Fries", "category": "Fast Food", "carb_range": (35, 45), "calorie_range": (250, 300), "protein_range": (3, 5), "fat_range": (10, 15), "gi_range": (75, 85)},
            {"name": "Doner Kebab", "category": "Fast Food", "carb_range": (25, 35), "calorie_range": (350, 400), "protein_range": (15, 20), "fat_range": (15, 20), "gi_range": (45, 55)},
            {"name": "Fish and Chips", "category": "Fast Food", "carb_range": (40, 50), "calorie_range": (400, 450), "protein_range": (12, 18), "fat_range": (20, 25), "gi_range": (60, 70)},
            {"name": "Chicken Nuggets", "category": "Fast Food", "carb_range": (10, 20), "calorie_range": (250, 300), "protein_range": (10, 15), "fat_range": (15, 20), "gi_range": (40, 50)},
            {"name": "Black Forest Cake", "category": "Cake", "carb_range": (40, 50), "calorie_range": (350, 400), "protein_range": (4, 7), "fat_range": (15, 20), "gi_range": (55, 65)},
            {"name": "Sacher Torte", "category": "Cake", "carb_range": (35, 45), "calorie_range": (300, 350), "protein_range": (5, 8), "fat_range": (15, 20), "gi_range": (50, 60)},
            {"name": "Cheesecake", "category": "Cake", "carb_range": (30, 40), "calorie_range": (300, 350), "protein_range": (6, 9), "fat_range": (20, 25), "gi_range": (45, 55)},
            {"name": "Carrot Cake", "category": "Cake", "carb_range": (35, 45), "calorie_range": (300, 350), "protein_range": (4, 7), "fat_range": (15, 20), "gi_range": (50, 60)},
            {"name": "Red Velvet Cake", "category": "Cake", "carb_range": (40, 50), "calorie_range": (350, 400), "protein_range": (4, 7), "fat_range": (15, 20), "gi_range": (55, 65)},
            {"name": "Gyros", "category": "Fast Food", "carb_range": (25, 35), "calorie_range": (300, 350), "protein_range": (12, 18), "fat_range": (12, 18), "gi_range": (45, 55)},
            {"name": "Fried Chicken Sandwich", "category": "Fast Food", "carb_range": (30, 40), "calorie_range": (350, 400), "protein_range": (12, 18), "fat_range": (15, 20), "gi_range": (50, 60)},
            {"name": "Falafel", "category": "Fast Food", "carb_range": (30, 40), "calorie_range": (250, 300), "protein_range": (6, 10), "fat_range": (10, 15), "gi_range": (50, 60)},
            {"name": "Bratwurst", "category": "Fast Food", "carb_range": (5, 15), "calorie_range": (250, 300), "protein_range": (10, 15), "fat_range": (15, 20), "gi_range": (40, 50)},
            {"name": "Currywurst", "category": "Fast Food", "carb_range": (10, 20), "calorie_range": (300, 350), "protein_range": (10, 15), "fat_range": (15, 20), "gi_range": (45, 55)},
            {"name": "Apple Strudel", "category": "Cake", "carb_range": (40, 50), "calorie_range": (300, 350), "protein_range": (4, 7), "fat_range": (12, 18), "gi_range": (55, 65)},
            {"name": "Baklava", "category": "Dessert", "carb_range": (35, 45), "calorie_range": (300, 350), "protein_range": (4, 7), "fat_range": (15, 20), "gi_range": (60, 70)},
            {"name": "Stollen", "category": "Cake", "carb_range": (40, 50), "calorie_range": (350, 400), "protein_range": (5, 8), "fat_range": (15, 20), "gi_range": (55, 65)},
            {"name": "Panettone", "category": "Cake", "carb_range": (45, 55), "calorie_range": (300, 350), "protein_range": (5, 8), "fat_range": (10, 15), "gi_range": (50, 60)},
            {"name": "Bienenstich", "category": "Cake", "carb_range": (35, 45), "calorie_range": (300, 350), "protein_range": (5, 8), "fat_range": (15, 20), "gi_range": (50, 60)},
            {"name": "Ciabatta", "category": "Bread", "carb_range": (50, 60), "calorie_range": (250, 300), "protein_range": (8, 12), "fat_range": (1, 3), "gi_range": (70, 80)},
            {"name": "Focaccia", "category": "Bread", "carb_range": (45, 55), "calorie_range": (250, 300), "protein_range": (7, 10), "fat_range": (5, 8), "gi_range": (65, 75)},
            {"name": "Sourdough Bread", "category": "Bread", "carb_range": (50, 60), "calorie_range": (200, 250), "protein_range": (6, 9), "fat_range": (1, 3), "gi_range": (50, 60)},
            {"name": "Rye Bread", "category": "Bread", "carb_range": (45, 55), "calorie_range": (200, 250), "protein_range": (6, 9), "fat_range": (1, 3), "gi_range": (50, 60)},
            {"name": "Brioche", "category": "Pastry", "carb_range": (40, 50), "calorie_range": (300, 350), "protein_range": (6, 9), "fat_range": (15, 20), "gi_range": (60, 70)},
            {"name": "Pain au Chocolat", "category": "Pastry", "carb_range": (40, 50), "calorie_range": (350, 400), "protein_range": (6, 9), "fat_range": (20, 25), "gi_range": (65, 75)},
            {"name": "Pumpernickel", "category": "Bread", "carb_range": (40, 50), "calorie_range": (180, 220), "protein_range": (5, 8), "fat_range": (1, 3), "gi_range": (45, 55)},
            {"name": "Danish Pastry", "category": "Pastry", "carb_range": (35, 45), "calorie_range": (300, 350), "protein_range": (5, 8), "fat_range": (15, 20), "gi_range": (60, 70)},
            {"name": "Borscht", "category": "Soup", "carb_range": (10, 20), "calorie_range": (80, 120), "protein_range": (3, 6), "fat_range": (2, 5), "gi_range": (40, 50)},
            {"name": "Spaghetti Bolognese", "category": "Main Dish", "carb_range": (60, 70), "calorie_range": (350, 400), "protein_range": (15, 20), "fat_range": (10, 15), "gi_range": (45, 55)},
            {"name": "Beef Wellington", "category": "Meat Dish", "carb_range": (20, 30), "calorie_range": (300, 350), "protein_range": (20, 25), "fat_range": (15, 20), "gi_range": (40, 50)},
            {"name": "Coq au Vin", "category": "Main Dish", "carb_range": (10, 20), "calorie_range": (200, 250), "protein_range": (15, 20), "fat_range": (8, 12), "gi_range": (40, 50)},
            {"name": "Moussaka", "category": "Main Dish", "carb_range": (20, 30), "calorie_range": (250, 300), "protein_range": (12, 18), "fat_range": (12, 18), "gi_range": (45, 55)},
            {"name": "Pierogi", "category": "Main Dish", "carb_range": (40, 50), "calorie_range": (200, 250), "protein_range": (6, 10), "fat_range": (5, 8), "gi_range": (50, 60)},
            {"name": "Churros", "category": "Dessert", "carb_range": (35, 45), "calorie_range": (250, 300), "protein_range": (3, 6), "fat_range": (10, 15), "gi_range": (60, 70)},
            {"name": "Crème Brûlée", "category": "Dessert", "carb_range": (20, 30), "calorie_range": (250, 300), "protein_range": (5, 8), "fat_range": (15, 20), "gi_range": (50, 60)},
            {"name": "Rösti", "category": "Side Dish", "carb_range": (20, 30), "calorie_range": (150, 200), "protein_range": (2, 4), "fat_range": (5, 8), "gi_range": (70, 80)},
            {"name": "Sauerkraut", "category": "Side Dish", "carb_range": (5, 10), "calorie_range": (40, 60), "protein_range": (1, 3), "fat_range": (0, 2), "gi_range": (30, 40)},
            {"name": "Kaiser Roll", "category": "Bread", "carb_range": (45, 55), "calorie_range": (200, 250), "protein_range": (6, 9), "fat_range": (2, 5), "gi_range": (65, 75)},
            {"name": "Baba au Rhum", "category": "Cake", "carb_range": (35, 45), "calorie_range": (300, 350), "protein_range": (4, 7), "fat_range": (10, 15), "gi_range": (55, 65)},
            {"name": "Opera Cake", "category": "Cake", "carb_range": (35, 45), "calorie_range": (350, 400), "protein_range": (5, 8), "fat_range": (15, 20), "gi_range": (50, 60)}
        ]

        all_foods = ethiopian_foods + european_foods
        data = {
            "Food_Name": [],
            "Category": [],
            "Carbohydrate_g_per_100g": [],
            "Glucose_g_per_100g": [],
            "Glycemic_Index": [],
            "Glycemic_Load": [],
            "Calories_kcal_per_100g": [],
            "Protein_g_per_100g": [],
            "Fat_g_per_100g": []
        }

        logger.info(f"Generating dataset with {n_samples} samples...")
        for _ in range(n_samples):
            food = random.choice(all_foods)
            carb_content = round(random.uniform(food["carb_range"][0], food["carb_range"][1]), 2)
            gi = round(random.uniform(food["gi_range"][0], food["gi_range"][1]), 2)
            # Estimate glucose content (simplified as proportion of carbs based on GI)
            glucose_content = round(carb_content * (gi / 100), 2)
            # Calculate glycemic load for 100g serving
            glycemic_load = round((carb_content * gi) / 100, 2)

            data["Food_Name"].append(food["name"].lower())
            data["Category"].append(food["category"])
            data["Carbohydrate_g_per_100g"].append(carb_content)
            data["Glucose_g_per_100g"].append(glucose_content)
            data["Glycemic_Index"].append(gi)
            data["Glycemic_Load"].append(glycemic_load)
            data["Calories_kcal_per_100g"].append(round(random.uniform(food["calorie_range"][0], food["calorie_range"][1]), 2))
            data["Protein_g_per_100g"].append(round(random.uniform(food["protein_range"][0], food["protein_range"][1]), 2))
            data["Fat_g_per_100g"].append(round(random.uniform(food["fat_range"][0], food["fat_range"][1]), 2))

        df = pd.DataFrame(data)
        logger.info("Dataset generated successfully.")
        return df

    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        raise

if __name__ == "__main__":
    df = generate_food_dataset()
    df.to_csv("../food_carbohydrate_dataset.csv", index=False)
    logger.info("Dataset saved to '../food_carbohydrate_dataset.csv'")