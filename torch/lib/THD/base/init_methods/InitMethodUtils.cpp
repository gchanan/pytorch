#include "InitMethodUtils.hpp"

#include <unistd.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <ifaddrs.h>

#include <tuple>
#include <iostream>

namespace thd {

namespace {

void sendPeerName(int socket) {
  struct sockaddr_storage master_addr;
  socklen_t master_addr_len = sizeof(master_addr);
  SYSCHECK(getpeername(socket, reinterpret_cast<struct sockaddr*>(&master_addr), &master_addr_len));

  std::string addr_str = sockaddrToString(reinterpret_cast<struct sockaddr*>(&master_addr));
  std::cerr << "sending addr_str " << addr_str << std::endl;
  send_string(socket, addr_str);
}

}

std::vector<std::string> getInterfaceAddresses() {
  struct ifaddrs *ifa;
  SYSCHECK(getifaddrs(&ifa));
  ResourceGuard ifaddrs_guard([ifa]() { ::freeifaddrs(ifa); });

  std::vector<std::string> addresses;

  while (ifa != nullptr) {
    struct sockaddr *addr = ifa->ifa_addr;
    if (addr) {
      bool is_loopback = ifa->ifa_flags & IFF_LOOPBACK;
      bool is_ip = addr->sa_family == AF_INET || addr->sa_family == AF_INET6;
      if (is_ip && !is_loopback) {
        addresses.push_back(sockaddrToString(addr));
      }
    }
    ifa = ifa->ifa_next;
  }

  return addresses;
}

std::string discoverWorkers(int listen_socket, rank_type world_size) {
  // accept connections from workers so they can know our address
  std::vector<int> sockets(world_size - 1);
  for (rank_type i = 0; i < world_size - 1; ++i) {
    std::tie(sockets[i], std::ignore) = accept(listen_socket);
  }

  std::string public_addr;
  for (auto socket : sockets) {
    std::cerr << "Sending name from master to workers" << std::endl;
    sendPeerName(socket);
    public_addr = recv_string(socket);
    ::close(socket);
  }
  return public_addr;
}

std::pair<std::string, std::string> discoverMaster(std::vector<std::string> addresses, port_type port) {
  // try to connect to address via any of the addresses
  std::string master_address = "";
  int socket;
  for (const auto& address : addresses) {
    try {
      std::cerr << "trying to connect with address " << address << std::endl;
      socket = connect(address, port, true, 2000);
      master_address = address;
      break;
    } catch (...) {} // when connection fails just try different address
  }

  if (master_address == "") {
    throw std::runtime_error("could not establish connection with other processes");
  }
  ResourceGuard socket_guard([socket]() { ::close(socket); });
  std::cerr << "Sending name from worker to master" << std::endl;
  sendPeerName(socket);
  std::string my_address = recv_string(socket);
  std::cerr << "master address " << master_address << " my address: " << my_address << std::endl;

  return std::make_pair(master_address, my_address);
}

}
