"""Gera um recorte .osm sintético em FORMATO OSM real: 2×2 cruzamentos
semaforizados (avenida principal × ruas), coordenadas na região da Paulista.
Serve para testar o pipeline da Fase 4 sem depender de download — o recorte
verdadeiro é extraído na máquina local (docs/PENDENTE_LOCAL.md)."""

from __future__ import annotations

from pathlib import Path

# ~300 m em graus na latitude de SP
DLAT = 0.0027
DLON = 0.00295 / 10.0  # ajustado: 0.00295 ≈ 300m / cos(23.5°)

LAT0, LON0 = -23.5600, -46.6550


def write_osm_fixture(path: Path) -> Path:
    dlat, dlon = DLAT, 0.00295
    lat = [LAT0, LAT0 - dlat]  # 2 ruas horizontais
    lon = [LON0, LON0 + dlon]  # 2 ruas verticais
    lat_stub = [LAT0 + dlat, LAT0 - 2 * dlat]
    lon_stub = [LON0 - dlon, LON0 + 2 * dlon]

    nodes: list[str] = []
    def node(nid: int, la: float, lo: float, signal: bool = False) -> None:
        if signal:
            nodes.append(
                f'  <node id="{nid}" lat="{la:.6f}" lon="{lo:.6f}" version="1">'
                f'<tag k="highway" v="traffic_signals"/></node>'
            )
        else:
            nodes.append(f'  <node id="{nid}" lat="{la:.6f}" lon="{lo:.6f}" version="1"/>')

    # cruzamentos (semaforizados): ids 11,12,21,22 -> (linha, coluna)
    node(11, lat[0], lon[0], signal=True)
    node(12, lat[0], lon[1], signal=True)
    node(21, lat[1], lon[0], signal=True)
    node(22, lat[1], lon[1], signal=True)
    # pontas das ruas horizontais (oeste/leste) e verticais (norte/sul)
    node(101, lat[0], lon_stub[0])
    node(102, lat[0], lon_stub[1])
    node(201, lat[1], lon_stub[0])
    node(202, lat[1], lon_stub[1])
    node(301, lat_stub[0], lon[0])
    node(302, lat_stub[1], lon[0])
    node(401, lat_stub[0], lon[1])
    node(402, lat_stub[1], lon[1])

    def way(wid: int, refs: list[int], highway: str, name: str) -> str:
        nds = "".join(f'<nd ref="{r}"/>' for r in refs)
        return (
            f'  <way id="{wid}" version="1">{nds}'
            f'<tag k="highway" v="{highway}"/><tag k="name" v="{name}"/>'
            f'<tag k="oneway" v="no"/></way>'
        )

    ways = [
        way(1001, [101, 11, 12, 102], "primary", "Avenida Fixture"),
        way(1002, [201, 21, 22, 202], "residential", "Rua Fixture Sul"),
        way(1003, [301, 11, 21, 302], "residential", "Rua Fixture Oeste"),
        way(1004, [401, 12, 22, 402], "residential", "Rua Fixture Leste"),
    ]

    content = (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        '<osm version="0.6" generator="traffic-rl-fixture">\n'
        + "\n".join(nodes) + "\n" + "\n".join(ways) + "\n</osm>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
