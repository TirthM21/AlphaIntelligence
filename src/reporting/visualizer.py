"""Chart generation utilities for AlphaIntelligence newsletters."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import os

class MarketVisualizer:
    """Generate professional financial charts for newsletters."""
    
    def __init__(self, output_dir: str = "./data/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set professional dark theme style
        plt.style.use('dark_background')
        self.colors = {
            'bg': '#0a0e27',
            'card': '#131836',
            'accent': '#c9a84c',
            'up': '#4ade80',
            'down': '#ef4444',
            'text': '#e0e0e0',
            'grid': '#1e2a5a'
        }

    def generate_sector_chart(self, sector_data: List[Dict]) -> str:
        """Create a bar chart of sector performance."""
        if not sector_data:
            return ""
            
        df = pd.DataFrame(sector_data)
        # Handle different FMP formats
        if 'changesPercentage' in df.columns:
            df['change'] = df['changesPercentage'].str.replace('%', '').astype(float)
        elif 'change' in df.columns:
            df['change'] = df['change'].astype(float)
            
        df = df.sort_values('change', ascending=True)
        
        plt.figure(figsize=(10, 6), facecolor=self.colors['bg'])
        ax = plt.gca()
        ax.set_facecolor(self.colors['bg'])
        
        colors = [self.colors['up'] if x > 0 else self.colors['down'] for x in df['change']]
        bars = plt.barh(df['sector'], df['change'], color=colors, alpha=0.8)
        
        plt.title('Daily Sector Performance (%)', color=self.colors['accent'], fontsize=14, pad=20)
        plt.grid(axis='x', linestyle='--', alpha=0.3, color=self.colors['grid'])
        
        # Add values on bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + (0.1 if width > 0 else -0.5)
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                     f'{width:+.2f}%', va='center', color=self.colors['text'], fontweight='bold')
            
        plt.tight_layout()
        output_path = self.output_dir / "sector_performance.png"
        plt.savefig(output_path, dpi=120, facecolor=self.colors['bg'])
        plt.close()
        return str(output_path)

    def generate_cap_comparison(self, cap_data: Dict[str, float]) -> str:
        """Compare Large, Mid, and Small cap performance."""
        if not cap_data:
            return ""
            
        labels = list(cap_data.keys())
        values = list(cap_data.values())
        
        plt.figure(figsize=(8, 4), facecolor=self.colors['bg'])
        ax = plt.gca()
        ax.set_facecolor(self.colors['bg'])
        
        colors = [self.colors['up'] if x > 0 else self.colors['down'] for x in values]
        bars = plt.bar(labels, values, color=colors, alpha=0.8, width=0.6)
        
        plt.title('Market Cap Segment Performance (%)', color=self.colors['accent'], fontsize=12)
        plt.ylabel('Change %', color=self.colors['text'])
        plt.axhline(0, color='white', linewidth=0.8)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + (0.05 if height > 0 else -0.15),
                     f'{height:+.2f}%', ha='center', color=self.colors['text'], fontweight='bold')
            
        plt.tight_layout()
        output_path = self.output_dir / "cap_comparison.png"
        plt.savefig(output_path, dpi=120, facecolor=self.colors['bg'])
        plt.close()
        return str(output_path)

    def generate_price_history(self, ticker: str, price_data: pd.DataFrame, signals: List[Dict] = None) -> str:
        """Generate a price history chart with signal annotations."""
        if price_data.empty:
            return ""
            
        plt.figure(figsize=(10, 5), facecolor=self.colors['bg'])
        ax = plt.gca()
        ax.set_facecolor(self.colors['bg'])
        
        # Plot Close price
        plt.plot(price_data.index, price_data['Close'], color=self.colors['text'], linewidth=1.5, label='Price')
        
        # Add SMA 50/200 if they exist
        if 'SMA_50' in price_data.columns:
            plt.plot(price_data.index, price_data['SMA_50'], color='#60a5fa', alpha=0.6, label='50 SMA')
        if 'SMA_200' in price_data.columns:
            plt.plot(price_data.index, price_data['SMA_200'], color='#f59e0b', alpha=0.6, label='200 SMA')
            
        plt.title(f'{ticker} Technical Analysis', color=self.colors['accent'], fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.2, color=self.colors['grid'])
        plt.legend(facecolor=self.colors['card'], edgecolor=self.colors['grid'])
        
        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        output_path = self.output_dir / f"{ticker}_history.png"
        plt.savefig(output_path, dpi=120, facecolor=self.colors['bg'])
        plt.close()
        return str(output_path)
