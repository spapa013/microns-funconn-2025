#!/bin/bash
# filepath: /src/microns-funconn-2025/show-registry.sh
#!/bin/bash
# Determine the registry name from git remote

# Get the origin remote URL
REMOTE_URL=$(git remote get-url origin 2>/dev/null)

if [ -z "$REMOTE_URL" ]; then
    # No git repo or no origin remote
    REGISTRY_NAME="cajal"
else
    # Handle different URL formats:
    # - https://github.com/username/repo.git
    # - git@github.com:username/repo.git
    # - git://github.com/username/repo.git
    if [[ $REMOTE_URL == *"github.com"* ]]; then
        if [[ $REMOTE_URL == *"github.com:"* ]]; then
            # SSH format
            REGISTRY_NAME=$(echo $REMOTE_URL | sed -E 's|.*github.com:([^/]+)/.*|\1|')
        else
            # HTTPS or git protocol
            REGISTRY_NAME=$(echo $REMOTE_URL | sed -E 's|.*github.com/([^/]+)/.*|\1|')
        fi
        
        # Convert to lowercase as GitHub Container Registry requires lowercase
        REGISTRY_NAME=$(echo $REGISTRY_NAME | tr '[:upper:]' '[:lower:]')
    else
        # Not a GitHub repository, defaulting to 'cajal'
        REGISTRY_NAME="cajal"
    fi
fi

# Print the registry name
echo "$REGISTRY_NAME"