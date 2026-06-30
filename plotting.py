import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def setup_plot_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['grid.color'] = '#EAEAEA'
    plt.rcParams['grid.linewidth'] = 0.6
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.edgecolor'] = '#EAEAEA'
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

def save_stage1_plots(history_file_or_data, output_dir):
    """
    Plots Stage 1 progressive generator and discriminator losses.
    """
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(history_file_or_data, str):
        if not os.path.exists(history_file_or_data):
            return
        df = pd.read_csv(history_file_or_data)
    else:
        df = pd.DataFrame(history_file_or_data)
        
    if df.empty:
        return
        
    epochs = df['Epoch']
    
    # Plot stage1_generator_losses.png
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if 'Loss_G1' in df.columns:
        ax.plot(epochs, df['Loss_G1'], label='loss2_1', color='#1f77b4', linewidth=1.8)
    if 'Loss_G2' in df.columns:
        ax.plot(epochs, df['Loss_G2'], label='loss2_2', color='#ff7f0e', linewidth=1.8)
        
    ax.set_title('Stage 1 Generator Progressive Pre-training Losses', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Pre-training Epoch', fontsize=10)
    ax.set_ylabel('Loss Value', fontsize=10)
    ax.grid(True)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stage1_generator_losses.png'), bbox_inches='tight')
    plt.close()

def save_stage2_plots(history_file_or_data, output_dir):
    """
    Plots Stage 2 training losses (total, classifier, and clip loss) and target domain OA.
    """
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(history_file_or_data, str):
        if not os.path.exists(history_file_or_data):
            return
        df = pd.read_csv(history_file_or_data)
    else:
        df = pd.DataFrame(history_file_or_data)
        
    if df.empty:
        return
        
    epochs = df['Epoch']
    
    # 1. Plot stage2_training_losses.png (Total Loss, Classification Loss)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if 'Loss_Total' in df.columns:
        ax.plot(epochs, df['Loss_Total'], label='Total Loss (Cls + $\lambda_{clip}$Clip)', color='#d62728', linewidth=1.8)
    if 'Loss_Cls' in df.columns:
        ax.plot(epochs, df['Loss_Cls'], label='Classifier Loss (CE)', color='#1f77b4', linewidth=1.5, linestyle='--')
        
    ax.set_title('Stage 2 Classifier Training Losses', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Training Epoch', fontsize=10)
    ax.set_ylabel('Loss Value', fontsize=10)
    ax.grid(True)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stage2_training_losses.png'), bbox_inches='tight')
    plt.close()
    
    # 2. Plot clip_loss_vs_epoch.png
    if 'Loss_Clip' in df.columns:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(epochs, df['Loss_Clip'], label='CLIP Semantic Loss', color='#9467bd', linewidth=1.8)
        ax.set_title('Stage 2 CLIP Semantic Alignment Loss', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Training Epoch', fontsize=10)
        ax.set_ylabel('CLIP Loss Value', fontsize=10)
        ax.grid(True)
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'clip_loss_vs_epoch.png'), bbox_inches='tight')
        plt.close()
        
    # 3. Plot oa_vs_epoch.png
    if 'OA' in df.columns:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(epochs, df['OA'], label='Target Domain OA (%)', color='#2ca02c', linewidth=2.0)
        ax.set_title('Stage 2 Target Domain Classification Accuracy', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Training Epoch', fontsize=10)
        ax.set_ylabel('Overall Accuracy (OA %)', fontsize=10)
        ax.grid(True)
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'oa_vs_epoch.png'), bbox_inches='tight')
        plt.close()
