#pragma once

#include <stdint.h>

typedef enum {
  SWARM_ROLE_FRONT_RANGER = 1,
  SWARM_ROLE_BACK_RANGER = 2,
  SWARM_ROLE_AI_STREAM = 3,
} swarm_role_t;

typedef enum {
  SWARM_CMD_NONE = 0,
  SWARM_CMD_HEARTBEAT = 1,
  SWARM_CMD_HOVER = 2,
  SWARM_CMD_LAND = 3,
  SWARM_CMD_STOP = 4,
} swarm_cmd_t;

typedef struct __attribute__((packed)) {
  uint16_t seq;
  uint8_t cmd;
  int16_t arg1;
  int16_t arg2;
  int16_t arg3;
  uint8_t flags;
} swarm_compact_command_t;

void swarmProtocolInit(void);
void swarmProtocolAcceptCommand(const swarm_compact_command_t *command);
uint8_t swarmProtocolHeartbeatOk(uint32_t nowMs, uint32_t timeoutMs);
uint16_t swarmProtocolLastSeq(void);
uint8_t swarmProtocolLastCmd(void);

