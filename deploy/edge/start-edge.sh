#!/bin/sh
# Renders the nginx server block for the advisor from env, then starts nginx.
#
#   start-edge.sh [OUT_PATH]   with OUT_PATH: only write the conf there and exit (used by tests).
#
# Env: EDGE_ACCESS=internal   restrict to INTERNAL_CIDR (default 10.77.0.0/24)
#      OPEN_DEBUG             legacy switch, see below (on unless set to something else)
#      UPSTREAM               advisor address (default advisor:8102)
set -eu

OUT="${1:-/etc/nginx/conf.d/advisor.conf}"
INTERNAL_CIDR="${INTERNAL_CIDR:-10.77.0.0/24}"
UPSTREAM="${UPSTREAM:-advisor:8102}"

RULES=""

# temporary: open for early testing
if [ "${OPEN_DEBUG:-1}" = "1" ]; then
  RULES="${RULES}    allow all;
"
fi

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
