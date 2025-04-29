import gym
from gym import spaces
import numpy as np

# Define a custom Gym environment

region_map = {'Indian': 0, 'Western': 1, 'Chinese': 2}  # Add all possible regions
diet_map = {'Vegetarian': 0, 'Non-Vegetarian': 1, 'Vegan': 2}  # Add all possible diet types

class MealPlanEnv(gym.Env):
    def __init__(self, user_prefs, meal_df):
        super(MealPlanEnv, self).__init__()
        self.meal_df = meal_df
        self.user_prefs = user_prefs
        self.action_space = spaces.Discrete(len(meal_df))  # Each meal is an action
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(user_prefs),), dtype=np.float32
        )
        self.current_step = 0
        self.max_steps = 7  # e.g., 7-day meal plan
        self.selected_meals = []

    def reset(self):
        self.current_step = 0
        self.selected_meals = []
        return np.array(list(self.user_prefs.values()), dtype=np.float32)

    def step(self, action):
        meal = self.meal_df.iloc[action]
        self.selected_meals.append(meal)
        self.current_step += 1

        meal_region = region_map.get(meal['region'], -1)
        meal_diet = diet_map.get(meal['diet_type'], -1)

        calorie_diff = abs(meal['calories_per_serving'] - self.user_prefs['calories_required'])
        region_match = 1 if meal_region == self.user_prefs['region'] else 0
        diet_match = 1 if meal_diet == self.user_prefs['diet_type'] else 0
        budget_ok = 1 if meal['cost_per_serving_in_inr'] <= self.user_prefs['budget'] else 0

        reward = -calorie_diff + 5*region_match + 5*diet_match + 2*budget_ok

        done = self.current_step >= self.max_steps
        obs = np.array(list(self.user_prefs.values()), dtype=np.float32)
        info = {}

        return obs, reward, done, info

    def render(self, mode='human'):
        print(f"Selected meals: {self.selected_meals}")
