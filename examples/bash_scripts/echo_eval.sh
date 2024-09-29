#!/bin/bash

# Function to print colored and bordered text without exceeding terminal width
print_bordered() {
    local text="$*"
    local text_length="${#text}"
    local term_width=$(tput cols)

    # Maximum length the border can be
    local max_length=$((term_width > 0 ? term_width : 80))

    # Actual length of the border (minimum of text_length and max_length)
    local border_length=$((text_length < max_length ? text_length : max_length))

    # Generate the border
    local border=$(printf '%*s' "$border_length" | tr ' ' '-')

    # Print the bordered text with color
    echo -e "\033[1;34m$border\033[0m"
    echo -e "\033[1;32m${text}\033[0m"
    echo -e "\033[1;34m$border\033[0m"
}

# Check if arguments are provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [arguments]"
    exit 1
fi

# Print the command with color and border, ensuring border is within terminal width
print_bordered "$@"

# Execute the command
eval "$@"
