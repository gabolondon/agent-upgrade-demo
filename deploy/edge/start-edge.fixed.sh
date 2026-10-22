#!/bin/sh
# Fixed version: the legacy "open for early testing" block (and OPEN_DEBUG) is gone, so the
# only access rule is the one EDGE_ACCESS asks for.
#
#   start-edge.sh [OUT_PATH]   with OUT_PATH: only write the conf there and exit (used by tests).
set -eu

OUT="${1:-/etc/nginx/conf.d/advisor.conf}"
INTERNAL_CIDR="${INTERNAL_CIDR:-10.77.0.0/24}"
UPSTREAM="${UPSTREAM:-advisor:8102}"

RULES=""

if [ "${EDGE_ACCESS:-}" = "internal" ]; then
  RULES="${RULES}    allow ${INTERNAL_CIDR};
    deny all;
"
fi

mkdir -p "$(dirname "$OUT")"
cat > "$OUT" <<CONF
server {
  listen 80;
  location / {
${RULES}    proxy_pass http://${UPSTREAM};
    proxy_set_header Host \$host;
    proxy_buffering off;
  }
}
CONF

if [ $# -ge 1 ]; then
  exit 0
fi

rm -f /etc/nginx/conf.d/default.conf
exec nginx -g 'daemon off;'
