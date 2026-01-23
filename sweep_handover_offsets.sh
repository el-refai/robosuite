#!/bin/bash

# Helper function for floating point arithmetic using awk
calc() {
    awk "BEGIN {printf \"%.6f\", $1}"
}

# Range definitions
YELLOW_X_MIN=-0.2
YELLOW_X_MAX=0.2
YELLOW_Y_MIN=0.15
YELLOW_Y_MAX=0.6

DUCT_X_MIN=-0.2
DUCT_X_MAX=0.2
DUCT_Y_MIN=-0.6
DUCT_Y_MAX=-0.15

# Number of positions per tape (arranged in a grid)
NUM_POSITIONS=8

# Grid dimensions for 8 positions (4×2 grid: 4 in X, 2 in Y)
NUM_X=4
NUM_Y=2

# Ensure outputs directory exists
mkdir -p outputs

# Calculate step sizes for 4×2 grid
if [ $NUM_X -eq 1 ]; then
    YELLOW_X_STEP=0.0
    DUCT_X_STEP=0.0
else
    YELLOW_X_STEP=$(calc "($YELLOW_X_MAX - $YELLOW_X_MIN) / ($NUM_X - 1)")
    DUCT_X_STEP=$(calc "($DUCT_X_MAX - $DUCT_X_MIN) / ($NUM_X - 1)")
fi

if [ $NUM_Y -eq 1 ]; then
    YELLOW_Y_STEP=0.0
    DUCT_Y_STEP=0.0
else
    YELLOW_Y_STEP=$(calc "($YELLOW_Y_MAX - $YELLOW_Y_MIN) / ($NUM_Y - 1)")
    DUCT_Y_STEP=$(calc "($DUCT_Y_MAX - $DUCT_Y_MIN) / ($NUM_Y - 1)")
fi

# Total combinations: 8 yellow positions × 8 duct positions = 64
total_combos=$((NUM_POSITIONS * NUM_POSITIONS))
current_combo=0

echo "=========================================="
echo "Handover Offset Sweep"
echo "=========================================="
echo "Positions per tape: $NUM_POSITIONS (${NUM_X}×${NUM_Y} grid)"
echo "Total combinations: $total_combos (${NUM_POSITIONS} yellow positions × ${NUM_POSITIONS} duct positions)"
echo ""
echo "Yellow tape ranges:"
echo "  X: [$YELLOW_X_MIN, $YELLOW_X_MAX] (${NUM_X} positions)"
echo "  Y: [$YELLOW_Y_MIN, $YELLOW_Y_MAX] (${NUM_Y} positions)"
echo "Duct tape ranges:"
echo "  X: [$DUCT_X_MIN, $DUCT_X_MAX] (${NUM_X} positions)"
echo "  Y: [$DUCT_Y_MIN, $DUCT_Y_MAX] (${NUM_Y} positions)"
echo "=========================================="
echo ""

# Generate all yellow tape positions (4×2 = 8 positions)
yellow_positions=()
for i in $(seq 0 $((NUM_X - 1))); do
    yellow_x_offset=$(calc "$YELLOW_X_MIN + $i * $YELLOW_X_STEP")
    for j in $(seq 0 $((NUM_Y - 1))); do
        yellow_y_offset=$(calc "$YELLOW_Y_MIN + $j * $YELLOW_Y_STEP")
        
        # Yellow tape position: base + offset
        yellow_x=$(calc "$YELLOW_BASE_X + $yellow_x_offset")
        yellow_y=$(calc "$YELLOW_BASE_Y + $yellow_y_offset")
        
        yellow_positions+=("${yellow_x},${yellow_y},0.0")
    done
done

# Generate all duct tape positions (4×2 = 8 positions)
duct_positions=()
for i in $(seq 0 $((NUM_X - 1))); do
    duct_x_offset=$(calc "$DUCT_X_MIN + $i * $DUCT_X_STEP")
    for j in $(seq 0 $((NUM_Y - 1))); do
        duct_y_offset=$(calc "$DUCT_Y_MIN + $j * $DUCT_Y_STEP")
        
        # Duct tape position: base + offset
        duct_x=$(calc "$DUCT_BASE_X + $duct_x_offset")
        duct_y=$(calc "$DUCT_BASE_Y + $duct_y_offset")
        
        duct_positions+=("${duct_x},${duct_y},0.0")
    done
done

echo "Generated ${#yellow_positions[@]} yellow tape positions"
echo "Generated ${#duct_positions[@]} duct tape positions"
echo "Starting sweep..."
echo ""

# Sweep over all combinations (8 × 8 = 64)
MAX_RETRIES=3
for yellow_offset_str in "${yellow_positions[@]}"; do
    for duct_offset_str in "${duct_positions[@]}"; do
        current_combo=$((current_combo + 1))
        
        attempt=1
        success=false
        while [ $attempt -le $MAX_RETRIES ]; do
            if [ $attempt -gt 1 ]; then
                echo "[$current_combo/$total_combos] Attempt $attempt/$MAX_RETRIES for:"
            else
                echo "[$current_combo/$total_combos] Running with:"
            fi
            echo "  Yellow tape offset: [$yellow_offset_str]"
            echo "  Duct tape offset: [$duct_offset_str]"
            
            # Run the test using uv run
            venv/bin/python test_scripts/test_handover_step.py \
                --yellow_offset="$yellow_offset_str" \
                --duct_offset="$duct_offset_str"
            
            exit_code=$?
            if [ $exit_code -eq 0 ]; then
                echo "  ✓ Completed successfully"
                success=true
                break
            elif [ $exit_code -eq 2 ]; then
                echo "  ✗ Failed: Video length > 1 min (exit code $exit_code)"
            else
                echo "  ✗ Failed with exit code $exit_code"
            fi
            
            echo "  Retrying... ($attempt/$MAX_RETRIES)"
            attempt=$((attempt + 1))
            echo ""
        done

        if [ "$success" = false ]; then
            echo "  !!! Giving up after $MAX_RETRIES attempts"
        fi
        echo ""
    done
done

echo "=========================================="
echo "Sweep completed!"
echo "Total combinations run: $current_combo"
echo "Videos saved in: dataset/"
echo "=========================================="
