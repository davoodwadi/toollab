import matplotlib.pyplot as plt
import seaborn as sns

BASE_FIG_WIDTH = 3.5  
BASE_FIG_HEIGHT = 2.5 
APA_STYLE = {
        # Fonts: Try Times New Roman first, then fallback to standard serif
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman", "serif"],
        
        # Font Sizes (APA Standard)
        "font.size": 10,
        "axes.titlesize": 12,      # 12pt for axes titles
        "axes.labelsize": 12,      # 12pt for axis labels (x and y)
        "xtick.labelsize": 10,     # 10pt for tick labels
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        
        # Figure Size (Default to one-and-a-half column width: ~5x3.5 inches)
        # You can override this in specific plots with plt.figure(figsize=(width, height))
        "figure.figsize": (5, 3.5),
        
        # Line styling
        "lines.linewidth": 1.5,
        "axes.linewidth": 1.0,     # Thickness of the axes bounding box lines
        
        # Spines/Borders (Remove top and right borders)
        "axes.spines.top": False,
        "axes.spines.right": False,
        
        # Ticks (Point outwards from the plot area)
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.bottom": True,
        "ytick.left": True,
        
        # Legend styling (Remove the bounding box)
        "legend.frameon": False,
        
        # High resolution saving for publication
        # "figure.dpi": 150,         # High resolution for inline display in Jupyter
        "savefig.dpi": 600,        # 600 DPI for high-quality print
        "savefig.bbox": "tight",   # Prevent cutting off labels or legends
        "savefig.format": "png",   # Default save format
    }
def set_apa_style():
    """
    Applies APA-compliant styling to Matplotlib and Seaborn figures.
    Optimized for grayscale print publication in journals like JAMS.
    """
    
    # 1. Base Seaborn Style: White background, no grid
    sns.set_style("white")
    sns.set_context("paper")
    
    # 2. Set Grayscale Palette
    # "Greys_r" goes from dark to light gray, ensuring high contrast for bars/lines
    sns.set_palette("Greys_r")
    
    # 3. Apply Matplotlib rcParams for fonts, sizes, and borders
    plt.rcParams.update(APA_STYLE)

def get_figsize(nrows=1, ncols=1):
    """
    Returns a tuple (width, height) scaled by the number of subplots.
    Usage: fig, ax = plt.subplots(2, 3, figsize=get_figsize(2, 3))
    """
    return (BASE_FIG_WIDTH * ncols, BASE_FIG_HEIGHT * nrows)

def format_label(label):
    """
    Converts snake_case or standard labels into Title Case for APA format.
    Example: 'completion_tokens_total' -> 'Completion Tokens Total'
             'model_num_params_log10' -> 'Model Num Params Log10'
    """
    if not label:
        return ""
    # Replace underscores with spaces and apply Title Case
    return label.replace('_', ' ').title()

def format_axes_labels(fig):
    """
    Iterates through all axes in a figure and formats the X and Y labels to Title Case.
    Usage: format_axes_labels(plt.gcf()) or format_axes_labels(fig)
    """
    for ax in fig.get_axes():
        # Get current labels
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        
        # Set formatted labels
        if xlabel:
            ax.set_xlabel(format_label(xlabel))
        if ylabel:
            ax.set_ylabel(format_label(ylabel))
# If someone runs this script directly, it won't do anything unless imported
if __name__ == "__main__":
    print("This module provides an APA styling function for matplotlib/seaborn.")
    print("To use it, import it in your notebook and call set_apa_style():")
    print("from apa_style import set_apa_style")
    print("set_apa_style()")
