#!/usr/bin/env python3
"""
RAGAS LIVE MONITOR
==================
Run this in a separate terminal to watch your results populate in real-time.
"""

import json
import time
import os
import pandas as pd
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# CONFIGURATION
RESULTS_FILE = os.getenv("RESULTS_FILE", "batch_ragas_evaluation_results_ultra_simple.json")
REFRESH_RATE = 10  # Seconds

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_data():
    """Safely load JSON data, handling read collisions."""
    try:
        if not os.path.exists(RESULTS_FILE):
            return None
            
        with open(RESULTS_FILE, 'r') as f:
            # Try-catch for JSONDecodeError in case file is being written to exactly now
            try:
                data = json.load(f)
                return data.get("results", [])
            except json.JSONDecodeError:
                return None
    except Exception:
        return None

def main():
    print("Waiting for results file...")
    
    while True:
        results = load_data()
        
        if results:
            clear_screen()
            df = pd.DataFrame(results)
            
            # Basic Stats
            total_count = len(df)
            success_count = len(df[df['status'] == 'success'])
            error_count = len(df[df['status'] == 'error'])
            
            # Calculate Averages per Mode
            if success_count > 0:
                # Filter only successful runs for stats
                success_df = df[df['status'] == 'success'].copy()
                
                # Extract metrics into columns if they are in a dict
                # (Assuming structure is flat or we flatten it)
                # If metrics are in a 'metrics' dict column:
                if 'metrics' in success_df.columns:
                    metrics_df = pd.json_normalize(success_df['metrics'])
                    success_df = pd.concat([success_df.drop(['metrics'], axis=1).reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)

                # Group by mode
                mode_stats = success_df.groupby('mode')[['ragas_score', 'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean()
                mode_counts = success_df['mode'].value_counts()
                
                # Add count column
                mode_stats['count'] = mode_counts
                
                # Sort by RAGAS Score (Descending)
                mode_stats = mode_stats.sort_values('ragas_score', ascending=False)
            
            # --- DISPLAY DASHBOARD ---
            print(f"📊 RAGAS LIVE MONITOR")
            print(f"⏰ Last Update: {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)
            print(f"Total Processed: {total_count}")
            print(f"✅ Success:      {success_count}")
            print(f"❌ Errors:       {error_count}")
            print("-" * 60)
            
            if success_count > 0:
                print("\n🏆 CURRENT STANDINGS (Average Scores):")
                # Format the dataframe for pretty printing
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 1000)
                pd.set_option('display.float_format', '{:.4f}'.format)
                print(mode_stats)
                
                print("\n📝 LATEST EVALUATION:")
                latest = df.iloc[-1]
                print(f"File:   {latest.get('file', 'N/A')}")
                print(f"Mode:   {latest.get('mode', 'N/A')}")
                print(f"Status: {latest.get('status', 'N/A')}")
                if latest.get('status') == 'success':
                    print(f"Score:  {latest.get('ragas_score', 0):.4f}")
            else:
                print("\nWaiting for first successful evaluation...")

        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Reading file...")

        time.sleep(REFRESH_RATE)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped.")