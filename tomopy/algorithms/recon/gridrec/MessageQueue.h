/*-----------------------------------------------------------------------------
 * Copyright (c) 2013, UChicago Argonne, LLC
 * See LICENSE file.
 *---------------------------------------------------------------------------*/

#ifndef MESSAGE_QUEUE_H
#define MESSAGE_QUEUE_H

/*---------------------------------------------------------------------------*/

#include <boost/thread.hpp>
#include <queue>

/*---------------------------------------------------------------------------*/

   /*
    * Message Queue info
    */
struct MessageQueueInfo {
    unsigned long   capacity;
    unsigned long   maxMessageSize;
    unsigned long  *buf;
    char           *firstMessageSlot;
    char           *lastMessageSlot;
    volatile char  *inPtr;
    volatile char  *outPtr;
    unsigned long   slotSize;
    bool            full;
};

typedef struct MessageQueueInfo *MessageQueueId;

   /**
    * @brief Thread safe queue
    */
class MessageQueue
{

public:

   /**
    * Constructor.
    */
   MessageQueue();

   /**
    * Destructor.
    */
   ~MessageQueue();

   /**
    * Check if queue is empty.
    */
   MessageQueueId MessageQueueCreate(unsigned int capacity,unsigned int maxMessageSize);


   void MessageQueueDestroy(MessageQueueId pmsg);

   /**
    * Pop an item from the queue. If no item is available return false.
    */
   int MessageQueueTrySend(MessageQueueId pmsg, void *message, unsigned int size);

   /**
    * Pop an item from the queue. Block until an item is available.
    */
   int MessageQueueTryReceive(MessageQueueId pmsg, void *message, unsigned int size);

private:

   /**
     * Mutex variable
    */
   boost::mutex m_mutex;

   /**
    * Condition variable
    */
   boost::condition_variable m_cond;

};

/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
