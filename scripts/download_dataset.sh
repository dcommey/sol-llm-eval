#!/bin/bash
# Download SmartBugs-Curated dataset

set -e

echo "Downloading SmartBugs-Curated dataset..."

# Create data directory if it doesn't exist
mkdir -p data/raw

# Clone the repository
if [ -d "data/raw/smartbugs-curated" ]; then
    echo "Dataset already exists. Pulling latest changes..."
    cd data/raw/smartbugs-curated
    git pull
    cd ../../..
else
    git clone https://github.com/smartbugs/smartbugs-curated.git data/raw/smartbugs-curated
fi

echo "Dataset download complete!"
echo "Dataset location: data/raw/smartbugs-curated"
