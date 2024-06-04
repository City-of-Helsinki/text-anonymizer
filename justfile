
# Production deployment

imagepath := "./image.tgz"
server := "prem"
service := "anonymizer"
service_args := "uvicorn anonymizer_api_webapp:main_app  --host 0.0.0.0"
docker_args := "-d --restart unless-stopped --network net -e MODE=webapi" # Args for Docker engine
platform := "linux/amd64"
commit := `git rev-parse --short HEAD`
image := "presidio-text-anonymizer:production"
image_sha := "presidio-text-anonymizer:" + commit

default: test

build:
    @echo "Building {{commit}} into container {{image}} and {{image_sha}}"
    docker build -t {{image}} --platform {{platform}} .
    docker tag {{image}} {{image_sha}}

test:
    echo "Just a test"

image-copy:
    docker image save {{image}} | DOCKER_HOST=ssh://{{server}} docker image load
    DOCKER_HOST=ssh://{{server}} docker tag {{image}} {{image_sha}}

publish:
    DOCKER_HOST=ssh://{{server}} docker stop {{service}} || true
    DOCKER_HOST=ssh://{{server}} docker rm {{service}} || true
    DOCKER_HOST=ssh://{{server}} docker run {{docker_args}} --name {{service}} {{image}} {{service_args}}
