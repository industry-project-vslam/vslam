import cflib.crtp

DEFAULT_RADIO_NETWORK = 0xE7E7E7E700
DEFAULT_RADIO_NETMASK = 0xFFFFFFFFF0
MAX_VALUE = 0xFFFFFFFFFF
EXCLUDE_BROADCAST = False # fasle = -0, true = -1
BROADCAST_OFFSET = int(EXCLUDE_BROADCAST)

def scan_drones(radio_network: int = DEFAULT_RADIO_NETWORK, radio_netmask: int = DEFAULT_RADIO_NETMASK) -> list[str]:
    """
    Scan all addresses in the radio network and return discovered interfaces.

    Returns:
        list of uris: list[str]
    """

    hosts = []
    max_hosts = ((~radio_netmask) & MAX_VALUE) - BROADCAST_OFFSET

    for offset in range(max_hosts):
        address = radio_network + offset + 1
        available = cflib.crtp.scan_interfaces(address)
        if available:
            hosts.extend(available)

    hosts = [host[0] for host in hosts]

    return hosts