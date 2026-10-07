#!/bin/sh
i=0
while :; do
    if curl -sf http://127.0.0.1:7125/printer/info | grep -Eq '"state"[[:space:]]*:[[:space:]]*"(ready|jogging)"'; then
        exit 0
    fi
    i=$((i + 1))
    if [ "$i" -gt 75 ]; then
        echo "klippy not ready after 150s" >&2
        exit 1
    fi
    sleep 2
done
