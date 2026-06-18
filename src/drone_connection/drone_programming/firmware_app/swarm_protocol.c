#include "swarm_protocol.h"

static volatile uint16_t lastSeq = 0;
static volatile uint8_t lastCmd = SWARM_CMD_NONE;
static volatile uint32_t lastHeartbeatMs = 0;

void swarmProtocolInit(void)
{
  lastSeq = 0;
  lastCmd = SWARM_CMD_NONE;
  lastHeartbeatMs = 0;
}

void swarmProtocolAcceptCommand(const swarm_compact_command_t *command)
{
  if (command == 0) {
    return;
  }

  lastSeq = command->seq;
  lastCmd = command->cmd;

  if (command->cmd == SWARM_CMD_HEARTBEAT) {
    /* The app layer should set this from the platform tick before calling. */
    lastHeartbeatMs = (uint32_t)command->arg1;
  }
}

uint8_t swarmProtocolHeartbeatOk(uint32_t nowMs, uint32_t timeoutMs)
{
  if (lastHeartbeatMs == 0) {
    return 0;
  }
  return ((nowMs - lastHeartbeatMs) <= timeoutMs) ? 1 : 0;
}

uint16_t swarmProtocolLastSeq(void)
{
  return lastSeq;
}

uint8_t swarmProtocolLastCmd(void)
{
  return lastCmd;
}

