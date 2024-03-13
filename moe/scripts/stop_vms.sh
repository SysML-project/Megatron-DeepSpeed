#!/bin/bash

# Check if a configuration file path was provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_config_file>"
    exit 1
fi

# Get the configuration file path from the script arguments
CONFIG_FILE=$1

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Read resource group and VM names from the JSON file
RESOURCE_GROUP=$(jq -r '.resourceGroup' "$CONFIG_FILE")
VMS=$(jq -r '.vms[]' "$CONFIG_FILE")

# Loop through the list of VMs and start each one
for VM_NAME in $VMS; do
    echo "Stopping $VM_NAME in resource group $RESOURCE_GROUP..."
    az vm deallocate --name "$VM_NAME" --resource-group "$RESOURCE_GROUP"
    if [ $? -eq 0 ]; then
        echo "$VM_NAME stopped successfully."
    else
        echo "Failed to stop $VM_NAME."
    fi
done