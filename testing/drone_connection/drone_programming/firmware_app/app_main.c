#include <stdint.h>

#include "FreeRTOS.h"
#include "task.h"
#include "log.h"
#include "param.h"

#include "swarm_protocol.h"

#define SWARM_HEARTBEAT_TIMEOUT_MS 1000

static uint8_t swarmRole = SWARM_ROLE_AI_STREAM;
static uint8_t swarmPassive = 1;
static uint8_t swarmEmergency = 0;
static uint8_t swarmState = 0;
static uint16_t swarmSeq = 0;
static uint8_t swarmLastCmd = SWARM_CMD_NONE;

static void swarmAppTask(void *param)
{
  (void)param;
  swarmProtocolInit();

  while (1) {
    swarmSeq = swarmProtocolLastSeq();
    swarmLastCmd = swarmProtocolLastCmd();

    if (!swarmPassive) {
      /* MVP is passive. Keep this branch empty until an app-layer executor is reviewed. */
    }

    if (swarmEmergency) {
      swarmState = 2;
    } else {
      swarmState = 1;
    }

    vTaskDelay(M2T(100));
  }
}

void appMain(void)
{
  xTaskCreate(swarmAppTask, "swarm_app", configMINIMAL_STACK_SIZE, 0, 1, 0);
}

PARAM_GROUP_START(swarm)
PARAM_ADD(PARAM_UINT8, role, &swarmRole)
PARAM_ADD(PARAM_UINT8, passive, &swarmPassive)
PARAM_ADD(PARAM_UINT8, emergency, &swarmEmergency)
PARAM_GROUP_STOP(swarm)

LOG_GROUP_START(swarm)
LOG_ADD(LOG_UINT8, state, &swarmState)
LOG_ADD(LOG_UINT16, seq, &swarmSeq)
LOG_ADD(LOG_UINT8, role, &swarmRole)
LOG_ADD(LOG_UINT8, lastCmd, &swarmLastCmd)
LOG_ADD(LOG_UINT8, emergency, &swarmEmergency)
LOG_GROUP_STOP(swarm)

