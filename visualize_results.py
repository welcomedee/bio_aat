"""
Visualization Script for Neuro-Swarm vs. Backpropagation Results
================================================================
This script generates comprehensive visualizations comparing the performance
of standard backpropagation vs. PSO-optimized neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Loading results...")
results = np.load('results.npy', allow_pickle=True).item()

# Extract data
baseline_train_acc = results['baseline_train_acc']
baseline_test_acc = results['baseline_test_acc']
baseline_train_time = results['baseline_train_time']
pso_train_acc = results['pso_train_acc']
pso_test_acc = results['pso_test_acc']
pso_train_time = results['pso_train_time']
pso_cost_history = results['pso_cost_history']
baseline_pred = results['baseline_pred']
pso_pred = results['pso_pred']
y_test = results['y_test']

# Create a comprehensive figure with multiple subplots
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Neuro-Swarm vs. Backpropagation: Comprehensive Comparison', 
             fontsize=16, fontweight='bold', y=0.995)

# --- Plot 1: Accuracy Comparison ---
ax1 = plt.subplot(2, 3, 1)
methods = ['Backpropagation', 'PSO (Nature-Inspired)']
train_accs = [baseline_train_acc * 100, pso_train_acc * 100]
test_accs = [baseline_test_acc * 100, pso_test_acc * 100]

x = np.arange(len(methods))
width = 0.35

bars1 = ax1.bar(x - width/2, train_accs, width, label='Training Accuracy', 
                color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width/2, test_accs, width, label='Testing Accuracy', 
                color='#e74c3c', alpha=0.8)

ax1.set_ylabel('Accuracy (%)', fontweight='bold')
ax1.set_title('Accuracy Comparison', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.legend()
ax1.set_ylim([90, 100])
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# --- Plot 2: PSO Convergence History ---
ax2 = plt.subplot(2, 3, 2)
iterations = range(1, len(pso_cost_history) + 1)
ax2.plot(iterations, pso_cost_history, linewidth=2, color='#2ecc71')
ax2.set_xlabel('Iteration', fontweight='bold')
ax2.set_ylabel('Cost (1 - Accuracy)', fontweight='bold')
ax2.set_title('PSO Convergence History', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.fill_between(iterations, pso_cost_history, alpha=0.3, color='#2ecc71')

# Add annotation for final cost
final_cost = pso_cost_history[-1]
ax2.annotate(f'Final Cost: {final_cost:.4f}',
            xy=(len(pso_cost_history), final_cost),
            xytext=(len(pso_cost_history)*0.7, final_cost*1.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, fontweight='bold', color='red')

# --- Plot 3: Training Time Comparison ---
ax3 = plt.subplot(2, 3, 3)
times = [baseline_train_time, pso_train_time]
colors = ['#3498db', '#e74c3c']
bars = ax3.bar(methods, times, color=colors, alpha=0.8)
ax3.set_ylabel('Time (seconds)', fontweight='bold')
ax3.set_title('Training Time Comparison', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}s',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# --- Plot 4: Confusion Matrix - Backpropagation ---
ax4 = plt.subplot(2, 3, 4)
cm_baseline = confusion_matrix(y_test, baseline_pred)
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'],
            ax=ax4, cbar_kws={'label': 'Count'})
ax4.set_ylabel('True Label', fontweight='bold')
ax4.set_xlabel('Predicted Label', fontweight='bold')
ax4.set_title('Confusion Matrix: Backpropagation', fontweight='bold')

# --- Plot 5: Confusion Matrix - PSO ---
ax5 = plt.subplot(2, 3, 5)
cm_pso = confusion_matrix(y_test, pso_pred)
sns.heatmap(cm_pso, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'],
            ax=ax5, cbar_kws={'label': 'Count'})
ax5.set_ylabel('True Label', fontweight='bold')
ax5.set_xlabel('Predicted Label', fontweight='bold')
ax5.set_title('Confusion Matrix: PSO', fontweight='bold')

# --- Plot 6: Performance Metrics Summary ---
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Calculate additional metrics
from sklearn.metrics import precision_score, recall_score, f1_score

baseline_precision = precision_score(y_test, baseline_pred)
baseline_recall = recall_score(y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred)

pso_precision = precision_score(y_test, pso_pred)
pso_recall = recall_score(y_test, pso_pred)
pso_f1 = f1_score(y_test, pso_pred)

summary_text = f"""
PERFORMANCE SUMMARY
{'='*40}

Backpropagation (Baseline):
  • Test Accuracy:  {baseline_test_acc*100:.2f}%
  • Precision:      {baseline_precision*100:.2f}%
  • Recall:         {baseline_recall*100:.2f}%
  • F1-Score:       {baseline_f1*100:.2f}%
  • Training Time:  {baseline_train_time:.2f}s

PSO (Nature-Inspired):
  • Test Accuracy:  {pso_test_acc*100:.2f}%
  • Precision:      {pso_precision*100:.2f}%
  • Recall:         {pso_recall*100:.2f}%
  • F1-Score:       {pso_f1*100:.2f}%
  • Training Time:  {pso_train_time:.2f}s

{'='*40}
Improvement: {(pso_test_acc - baseline_test_acc)*100:+.2f}%

KEY ADVANTAGE:
PSO directly optimizes accuracy,
while backpropagation uses proxy
loss functions (cross-entropy).
"""

ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'comparison_results.png'")
plt.show()

# --- Additional Plot: Cost History with Detailed Analysis ---
fig2, ax = plt.subplots(figsize=(12, 6))
ax.plot(iterations, pso_cost_history, linewidth=2.5, color='#2ecc71', label='PSO Cost')
ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax.set_ylabel('Cost (1 - Accuracy)', fontsize=12, fontweight='bold')
ax.set_title('Particle Swarm Optimization: Convergence Analysis', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.fill_between(iterations, pso_cost_history, alpha=0.2, color='#2ecc71')

# Add markers for key points
min_cost_idx = np.argmin(pso_cost_history)
ax.plot(min_cost_idx + 1, pso_cost_history[min_cost_idx], 'r*', 
        markersize=15, label=f'Best Cost: {pso_cost_history[min_cost_idx]:.4f}')

# Add convergence annotation
ax.axhline(y=pso_cost_history[-1], color='r', linestyle='--', alpha=0.5)
ax.text(len(pso_cost_history)*0.5, pso_cost_history[-1]*1.1, 
        f'Converged to: {pso_cost_history[-1]:.4f}',
        fontsize=11, fontweight='bold', color='red',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('pso_convergence.png', dpi=300, bbox_inches='tight')
print("PSO convergence plot saved as 'pso_convergence.png'")
plt.show()

print("\n" + "="*60)
print("All visualizations generated successfully!")
print("="*60)
