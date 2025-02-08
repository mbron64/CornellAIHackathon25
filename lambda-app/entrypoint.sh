#!/bin/sh

if [ -z "${AWS_LAMBDA_RUNTIME_API}" ]
then

    ARCH=$(uname -m)

    if [ "${ARCH}" = "aarch64" ]
    then

        ARCH="arm64"

    fi

    curl -Lo /usr/local/bin/aws-lambda-rie "https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie-${ARCH}"
    chmod +x /usr/local/bin/aws-lambda-rie

    exec /usr/local/bin/aws-lambda-rie python -m awslambdaric $@

else

    exec python -m awslambdaric $@

fi
