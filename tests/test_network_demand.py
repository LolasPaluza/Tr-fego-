import xml.etree.ElementTree as ET

from traffic_rl.config import N_STAGES
from traffic_rl.envs.demand import build_routes, road_class_of_vehicle
from traffic_rl.envs.phases import build_phase_table


def test_network_builds(scenario_files):
    net_file, _ = scenario_files
    assert net_file.exists()
    root = ET.parse(net_file).getroot()
    edge_ids = {e.get("id") for e in root.iter("edge") if not e.get("function")}
    assert {"in_E", "in_W", "in_N", "in_S", "out_E", "out_W", "out_N", "out_S"} <= edge_ids


def test_phase_table_structure(scenario_files):
    net_file, _ = scenario_files
    table = build_phase_table(str(net_file))
    assert len(table.green_states) == N_STAGES
    n = table.n_links
    assert n > 0
    for green, yellow in zip(table.green_states, table.yellow_states, strict=True):
        assert len(green) == n and len(yellow) == n
        assert "G" in green, "todo estágio precisa liberar algum movimento"
        # amarelo cobre exatamente os links que eram verdes
        assert all((g == "G") == (y == "y") for g, y in zip(green, yellow, strict=True))
    assert table.all_red_state == "r" * n
    # nenhum link fica verde em dois estágios de vias conflitantes (avenida × local)
    av = set(i for i, c in enumerate(table.green_states[0]) if c == "G")
    local = set(i for i, c in enumerate(table.green_states[2]) if c == "G")
    assert not (av & local)


def test_routes_generated(cfg, scenario_files):
    _, route_file = scenario_files
    root = ET.parse(route_file).getroot()
    flows = root.findall("flow")
    # pico_assimetrico tem caminhões: 4 aproximações × 3 movimentos × 2 tipos
    assert len(flows) == 4 * 3 * 2
    total_vph = sum(float(f.get("probability")) * 3600 for f in flows)
    sc = cfg.scenario("pico_assimetrico")
    esperado = 2 * sc.avenue_flow_vph + 2 * sc.local_flow_vph
    assert abs(total_vph - esperado) < 1.0


def test_routes_without_trucks(cfg, tmp_path):
    route_file = build_routes(cfg.scenario("fora_pico"), tmp_path)
    root = ET.parse(route_file).getroot()
    flows = root.findall("flow")
    assert len(flows) == 4 * 3  # sem caminhões
    assert all(f.get("type") == "carro" for f in flows)


def test_road_class_from_vehicle_id():
    assert road_class_of_vehicle("E_s_carro.12") == "avenida"
    assert road_class_of_vehicle("W_l_caminhao.3") == "avenida"
    assert road_class_of_vehicle("N_r_carro.0") == "local"
    assert road_class_of_vehicle("S_s_carro.7") == "local"
