<template>
  <v-container class="fill-height">
    <v-responsive class="align-center text-center">
      <div class="chat-container">
        <div class="welcome-message" :class="{ 'bot-bubble': true }">
          <div class="text-body-2 font-weight-light mb-n1 white-text">Welcome to</div>
          <h1 class="text-h2 font-weight-bold white-text">CampusCompanion</h1>
        </div>
        <chatWindow :conversation="conversation" />
      </div>
      <div class="py-14" />
    </v-responsive>
    <v-text-field v-model="newMessageText" color="primary" label="Chat" variant="filled" class="chat-input"
      @keydown.enter="sendMessage">
      <template #append>
        <transition name="scale">
          <v-btn icon @click="sendMessage">
            <v-icon>mdi-send</v-icon>
          </v-btn>
        </transition>
      </template>
    </v-text-field>
  </v-container>
</template>

<style scoped>
.align-center {
  align-items: center;
}

.text-center {
  text-align: center;
}

.fill-height {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.chat-container {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  position: relative;
}

.welcome-message {
  background-color: #a1a0a0;
  border-radius: 15px;
  padding: 10px;
  margin-bottom: 10px;
  position: sticky;
  top: 0;
  z-index: 2;
}

.chat-input {
  width: 40vw;
  margin: auto;
  position: absolute;
  bottom: 10px;
  z-index: 1;
}

.chat-window {
  overflow-y: auto;
  flex-grow: 1;
  max-height: 70vh
}

/* Hide scrollbar for Chrome, Safari, and Opera */
.chat-window::-webkit-scrollbar {
  width: 0;
}


.message-container:hover .timestamp {
  opacity: 1;
  /* Show timestamp on hover */
}

.white-text {
  color: #fff;
  /* Set text color to white */
}

.scale-enter-active,
.scale-leave-active {
  transform: scale(1);
  transition: transform 0.2s;
}

.scale-enter,
.scale-leave-to {
  transform: scale(0.8);
}
</style>

<script setup>
import chatWindow from "./ChatWindow.vue";
import { ref } from "vue";
import axiosCom from "@/components/axios"

const messages = ref([]);
const newMessageText = ref("");
const conversation = ref({
  messages: [
    { is_bot: true, message: "Welcome to" },
    { is_bot: true, message: "CampusCompanion" },
    { is_bot: true, message: "Feel free to chat!" },
    { is_bot: true, message: "Ask me anything." },
    { is_bot: false, message: "Hello Back" },
    // Add more messages as needed
  ],
});
const error = ref(null);

const sendMessage = async () => {
  const message = {
    is_bot: false,
    message: newMessageText.value.trim(),
  };

  if (message.message !== "") {
    conversation.value.messages.push(message);

    // REST CALL
    newMessageText.value = "";
    try {
      console.log(message.message)
      // wait for axios to return
      const response = await axiosCom.post(`/chatbot/${message.message}`);
      console.log(response.data.message)
      const botMessage = {
        is_bot: true,
        message: response.data.message,
      };
      conversation.value.messages.push(botMessage);
      conversation.value.focus()
    } catch (err) {
      error.value = err.message;
    }
  }
};
</script>