#!/bin/bash
# Clean temporary and cache files from AlphaAgent project

echo "🧹 Cleaning AlphaAgent project..."
echo ""

# Remove Python cache
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null

# Remove macOS files
echo "Removing macOS files..."
find . -name ".DS_Store" -delete 2>/dev/null

# Remove log files (optional - uncomment if you want to clean logs)
# echo "Removing log files..."
# find . -name "*.log" -delete 2>/dev/null

# Remove temporary files
echo "Removing temporary files..."
find . -name "*~" -delete 2>/dev/null
find . -name "*.swp" -delete 2>/dev/null

# Show what remains
echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Project size after cleanup:"
du -sh . 2>/dev/null | awk '{print "  " $1}'

