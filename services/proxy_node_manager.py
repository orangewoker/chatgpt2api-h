"""Proxy Node Manager for per-account proxy routing and protocol conversion."""

from __future__ import annotations

import atexit
import base64
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from utils.log import logger

_LOCAL_PORT_START = 19100
_ACTIVE_CORES: dict[str, dict] = {}
_LOCK = threading.RLock()


def _is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) != 0


def _allocate_free_port() -> int:
    for port in range(_LOCAL_PORT_START, _LOCAL_PORT_START + 500):
        if port not in [item.get("port") for item in _ACTIVE_CORES.values()] and _is_port_free(port):
            return port
    return _LOCAL_PORT_START


def parse_vless_url(url: str) -> dict:
    """Parse vless:// link into structured parameters."""
    parsed = urlparse(url)
    uuid = parsed.username or ""
    server = parsed.hostname or ""
    port = parsed.port or 443
    params = parse_qs(parsed.query)

    security = params.get("security", ["none"])[0]
    flow = params.get("flow", [""])[0]
    network = params.get("type", ["tcp"])[0]
    sni = params.get("sni", [server])[0]
    pbk = params.get("pbk", [""])[0]
    sid = params.get("sid", [""])[0]
    path = params.get("path", ["/"])[0]
    service_name = params.get("serviceName", [""])[0]

    return {
        "protocol": "vless",
        "uuid": uuid,
        "server": server,
        "port": port,
        "security": security,
        "flow": flow,
        "network": network,
        "sni": sni,
        "public_key": pbk,
        "short_id": sid,
        "path": path,
        "service_name": service_name,
        "name": unquote(parsed.fragment or f"vless-{server}"),
    }


def parse_vmess_url(url: str) -> dict:
    """Parse vmess:// link (base64 JSON) into structured parameters."""
    raw = url[len("vmess://"):]
    # Fix padding
    missing_padding = len(raw) % 4
    if missing_padding:
        raw += "=" * (4 - missing_padding)
    try:
        decoded = base64.b64decode(raw).decode("utf-8")
        data = json.loads(decoded)
        return {
            "protocol": "vmess",
            "server": data.get("add", ""),
            "port": int(data.get("port", 443)),
            "uuid": data.get("id", ""),
            "alter_id": int(data.get("aid", 0)),
            "cipher": data.get("scy", "auto"),
            "network": data.get("net", "tcp"),
            "path": data.get("path", "/"),
            "host": data.get("host", ""),
            "tls": data.get("tls", "") == "tls",
            "sni": data.get("sni", data.get("host", "")),
            "name": data.get("ps", "vmess-node"),
        }
    except Exception as exc:
        logger.warning(f"Failed to decode vmess URL: {exc}")
        return {"protocol": "vmess", "raw": url}


def find_mihomo_executable() -> str | None:
    """Check for bundled or system mihomo executable."""
    candidates = [
        shutil.which("mihomo"),
        shutil.which("clash-meta"),
        shutil.which("xray"),
        # Bundled paths in common deployments
        str(Path(__file__).parent.parent / "bin" / "mihomo"),
        str(Path(__file__).parent.parent / "bin" / "mihomo.exe"),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


class ProxyNodeManager:
    """Manages protocol conversion and per-account proxy resolution."""

    @classmethod
    def resolve_account_proxy(cls, proxy_string: str) -> str:
        """Takes an account proxy string and returns a curl_cffi compatible proxy URL.

        Supports:
        - Direct: "" -> ""
        - SOCKS5/HTTP: "socks5://...", "http://..." -> "socks5h://...", "http://..."
        - VLESS / VMess: "vless://...", "vmess://..." -> converts or starts local SOCKS5 bridge
        """
        raw = str(proxy_string or "").strip()
        if not raw:
            return ""

        lower = raw.lower()
        if lower.startswith(("http://", "https://", "socks5://", "socks5h://", "socks4://")):
            if lower.startswith("socks5://"):
                return "socks5h://" + raw[len("socks5://"):]
            if lower.startswith("socks://"):
                return "socks5h://" + raw[len("socks://"):]
            return raw

        if lower.startswith("vless://") or lower.startswith("vmess://"):
            with _LOCK:
                if raw in _ACTIVE_CORES:
                    return f"socks5h://127.0.0.1:{_ACTIVE_CORES[raw]['port']}"

                # Attempt to bridge via local Mihomo if available
                local_bridge = cls._start_node_bridge_locked(raw)
                if local_bridge:
                    return local_bridge

            logger.warning(
                f"VLESS/VMess proxy configured ({raw[:30]}...) but no local Mihomo core was found. "
                f"If using an external client (v2rayN/Clash), please specify its local SOCKS5 port (e.g., socks5://127.0.0.1:10808)."
            )
            return raw

        return raw

    @classmethod
    def _start_node_bridge_locked(cls, node_url: str) -> str | None:
        exe = find_mihomo_executable()
        if not exe:
            return None

        port = _allocate_free_port()
        config_dir = Path("data/proxy_runtime")
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"node_{port}.yaml"

        # Generate minimal Mihomo config
        try:
            proxy_node = None
            if node_url.lower().startswith("vless://"):
                info = parse_vless_url(node_url)
                proxy_node = {
                    "name": "node",
                    "type": "vless",
                    "server": info["server"],
                    "port": info["port"],
                    "uuid": info["uuid"],
                    "network": info["network"],
                    "tls": info["security"] in {"tls", "reality"},
                    "udp": True,
                }
                if info.get("flow"):
                    proxy_node["flow"] = info["flow"]
                if info.get("sni"):
                    proxy_node["servername"] = info["sni"]
                if info["security"] == "reality":
                    proxy_node["reality-opts"] = {
                        "public-key": info["public_key"],
                        "short-id": info["short_id"],
                    }
            elif node_url.lower().startswith("vmess://"):
                info = parse_vmess_url(node_url)
                proxy_node = {
                    "name": "node",
                    "type": "vmess",
                    "server": info["server"],
                    "port": info["port"],
                    "uuid": info["uuid"],
                    "alterId": info.get("alter_id", 0),
                    "cipher": info.get("cipher", "auto"),
                    "network": info.get("network", "tcp"),
                    "tls": info.get("tls", False),
                    "udp": True,
                }
                if info.get("sni"):
                    proxy_node["servername"] = info["sni"]

            if not proxy_node:
                return None

            import yaml
            cfg = {
                "mixed-port": port,
                "allow-lan": False,
                "mode": "rule",
                "log-level": "silent",
                "proxies": [proxy_node],
                "rules": ["MATCH,node"],
            }
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f)

            proc = subprocess.Popen(
                [exe, "-f", str(config_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.3)

            _ACTIVE_CORES[node_url] = {
                "port": port,
                "proc": proc,
                "config_path": config_path,
            }

            return f"socks5h://127.0.0.1:{port}"
        except Exception as exc:
            logger.warning(f"Failed to spawn mihomo bridge for node: {exc}")
            return None

    @classmethod
    def shutdown_all(cls):
        with _LOCK:
            for item in _ACTIVE_CORES.values():
                try:
                    proc = item.get("proc")
                    if proc:
                        proc.terminate()
                    cfg = item.get("config_path")
                    if cfg and os.path.exists(cfg):
                        os.remove(cfg)
                except Exception:
                    pass


atexit.register(ProxyNodeManager.shutdown_all)
