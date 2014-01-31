/*-----------------------------------------------------------------------------
 * Copyright (c) 2013, UChicago Argonne, LLC
 * See LICENSE file.
 *---------------------------------------------------------------------------*/

#include <boost/thread.hpp>
#include <queue>
#include <MessageQueue.h>

/*---------------------------------------------------------------------------*/

MessageQueue::MessageQueue()
{

}

/*---------------------------------------------------------------------------*/

MessageQueue::~MessageQueue()
{

}

/*---------------------------------------------------------------------------*/

MessageQueueId MessageQueue::MessageQueueCreate(unsigned int capacity,unsigned int maxMessageSize)
{

   MessageQueueId pmsg;
   unsigned int slotBytes, slotLongs;

   assert(capacity != 0);

   pmsg = (MessageQueueId)calloc(1, sizeof(*pmsg));

   pmsg->capacity = capacity;
   pmsg->maxMessageSize = maxMessageSize;
   slotLongs = 1 + ((maxMessageSize + sizeof(unsigned long) - 1) / sizeof(unsigned long));
   slotBytes = slotLongs * sizeof(unsigned long);

   pmsg->buf = (unsigned long *)calloc(pmsg->capacity, slotBytes);

   pmsg->inPtr = pmsg->outPtr = pmsg->firstMessageSlot = (char *)&pmsg->buf[0];
   pmsg->lastMessageSlot = (char *)&pmsg->buf[(capacity - 1) * slotLongs];
   pmsg->full = false;
   pmsg->slotSize = slotBytes;

   return pmsg;

}

/*---------------------------------------------------------------------------*/

void MessageQueue::MessageQueueDestroy(MessageQueueId pmsg)
{

   free(pmsg->buf);
   free(pmsg);

}

/*---------------------------------------------------------------------------*/

int MessageQueue::MessageQueueTrySend(MessageQueueId pmsg, void *message, unsigned int size)
{

   boost::lock_guard<boost::mutex> lock(m_mutex);

   char *myInPtr, *nextPtr;

   if(size > pmsg->maxMessageSize)
       return -1;

   //Copy to queue

   myInPtr = (char *)pmsg->inPtr;
   if (myInPtr == pmsg->lastMessageSlot)
       nextPtr = pmsg->firstMessageSlot;
   else
       nextPtr = myInPtr + pmsg->slotSize;
   if (nextPtr == (char *)pmsg->outPtr)
       pmsg->full = true;
   *(volatile unsigned long *)myInPtr = size;
   memcpy((unsigned long *)myInPtr + 1, message, size);
   pmsg->inPtr = nextPtr;

   return 0;

}

/*---------------------------------------------------------------------------*/

int MessageQueue::MessageQueueTryReceive(MessageQueueId pmsg, void *message, unsigned int size)
{

   boost::lock_guard<boost::mutex> lock(m_mutex);

   char *myOutPtr;
   unsigned long l;

   //If there's a message on the queue, copy it
   myOutPtr = (char *)pmsg->outPtr;
   if ((myOutPtr != pmsg->inPtr) || pmsg->full) {
       int ret;
       l = *(unsigned long *)myOutPtr;
       if (l <= size) {
           memcpy(message, (unsigned long *)myOutPtr + 1, l);
           ret = l;
       }
       else {
           ret = -1;
       }
       if (myOutPtr == pmsg->lastMessageSlot)
           pmsg->outPtr = pmsg->firstMessageSlot;
       else
           pmsg->outPtr += pmsg->slotSize;
       pmsg->full = false;

       return ret;
   }
   else
   {
      return -1;
   }

}

/*---------------------------------------------------------------------------*/

