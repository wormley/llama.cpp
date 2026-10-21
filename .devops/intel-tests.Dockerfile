# intel-tests.Dockerfile — test-runner image on top of the intel.Dockerfile build stage.
# Carries test-backend-ops compiled with tests enabled, so GPU hosts can run backend-op
# numerics tests without building anything (pull from the registry and run).
#
# Build (on the build host, after `podman build --target build -t localhost/llama.cpp-intel:<tag>-build`):
#   podman build -t localhost/llama.cpp-intel:<tag>-tests -f .devops/intel-tests.Dockerfile .
#
# Run (GPU host):
#   podman run --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
#       <registry>/<img>:<tag>-tests -b SYCL0 -o 'DSV4_HC.*'
FROM localhost/llama.cpp-intel:hc-post-fusion-build

COPY .devops/intel-tests.Dockerfile /app/

RUN cmake -B /app/build -DLLAMA_BUILD_TESTS=ON && \
    cmake --build /app/build --target test-backend-ops -j"$(nproc)" && \
    printf '#!/bin/bash\nsource /opt/intel/oneapi/compiler/latest/env/vars.sh\nexec /app/build/bin/test-backend-ops "$@"\n' \
        > /usr/local/bin/test-backend-ops && \
    chmod +x /usr/local/bin/test-backend-ops

ENTRYPOINT ["/usr/local/bin/test-backend-ops"]
