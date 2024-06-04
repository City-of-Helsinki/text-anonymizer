#!/bin/bash
if [ $# -lt 1 ]; then
    echo "Please provide the correct number of arguments."
    echo "- build: deploy.sh build"
    echo "- deploy: deploy.sh deploy [username]"
    exit 1
fi

username=$2

# Variables
commit=$(git rev-parse --short HEAD)
image="presidio-text-anonymizer:production"
image_sha="presidio-text-anonymizer:$commit"
platform="linux/amd64" # Assuming platform is linux/amd64, update as needed

echo "Howto:  ./deploy build or ./deploy deploy [username]"

build() {
  echo "Disconnect from VPN before running this script"
  echo "Also consider running train_custom_spacy_model/train_custom_spacy_model.py before running this script"
  echo "Building $commit into container $image and $image_sha"
  docker build --no-cache -t $image --platform $platform .
  docker tag $image $image_sha
}

deploy() {
  if [ -z "$username" ]; then
    echo "Please provide the username as a command line argument for the deploy command."
    exit 1
  fi

  # Push image to prem
  echo "Connect to VPN before running this script"
  echo "Pushing $image and $image_sha to prem, please login and wait..."
  docker save $image | gzip | ssh $username@prem docker load

  echo "Ready to deploy $image to prem:"

  echo "1. Login to prem: ssh $username@prem"
  echo "2. Run: docker stop anonymizer"
  echo "3. Run: docker rm anonymizer"
  echo "4. Run: docker run -d --restart unless-stopped --network net -e MODE=webapi --name anonymizer $image"
  echo "5. Run: docker ps"
  echo "6. Check that the container is running: docker logs anonymizer"
}

other() {
    echo "run ./deploy build or ./deploy deploy [username]"
}

# Parse arguments and run appropriate function
for arg in "$@"
do
    case $arg in
        build)
            build
            ;;
        deploy)
            deploy
            ;;
        *)
            other
            ;;
    esac
done