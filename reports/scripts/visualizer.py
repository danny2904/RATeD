
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
import os
from sklearn.metrics import confusion_matrix

class Visualizer:
    def __init__(self, output_dir='reports/figures'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_confusion_matrix(self, y_true, y_pred, labels, filename, title='Confusion Matrix'):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        # Determine cmap based on filename or title?
        cmap = 'Reds' if 'hx' in filename else 'Blues'
        
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path

    def plot_wordcloud(self, tokens, filename, title='Toxic Spans'):
        text = " ".join(tokens)
        if not text:
            return None
        try:
            colormap = 'inferno' if 'hx' in filename else 'Reds'
            wc = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc.to_image(), interpolation='bilinear')
            plt.axis('off')
            plt.title(title)
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=300)
            plt.close()
            return save_path
        except Exception as e:
            print(f"WordCloud Logic Error: {e}")
            return None

    def plot_loss(self, history_path, filename, title='Training Loss'):
        if not os.path.exists(history_path):
            return None
            
        plt.figure(figsize=(8, 5))
        try:
            # Handle JSON vs CSV
            if history_path.endswith('.json'):
                import json
                with open(history_path, 'r') as f: history = json.load(f)
                plt.plot(history.get('train_loss', []), marker='o', label='Train')
                plt.plot(history.get('val_loss', []), marker='s', label='Val', linestyle='--')
            elif history_path.endswith('.csv'):
                df = pd.read_csv(history_path)
                plt.plot(df['Epoch'], df['Train_Loss'], marker='o', label='Train')
                plt.plot(df['Epoch'], df['Val_Loss'], marker='s', label='Val', linestyle='--')
                
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=300)
            plt.close()
            return save_path
        except Exception as e:
            print(f"Plot Loss Error: {e}")
            return None
