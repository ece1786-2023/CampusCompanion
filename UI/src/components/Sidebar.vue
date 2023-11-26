<!-- MainComponent.vue -->
<template>
  <v-card>
    <v-layout>
      <v-navigation-drawer
        v-model="drawer"
        :rail="rail"
        permanent
        @click="rail = false"
      >
        <v-list density="compact" nav>
          <v-list-item prepend-icon="mdi-restart" title="New Conversation" value="new_conv">
            <template v-slot:append>
              <v-btn
                variant="text"
                icon="mdi-chevron-left"
                @click.stop="rail = !rail"
              ></v-btn>
            </template>
          </v-list-item>
          <v-divider></v-divider>
          <v-list-item @click="uploadModal = true" prepend-icon="mdi-upload" title="Upload Transcript" value="account"></v-list-item>
        </v-list>
      </v-navigation-drawer>
      <v-main style="height: calc(100vh - 65px)">

       
        <v-dialog v-model="uploadModal" max-width="500">
          <v-card>
            <v-card-title></v-card-title>
                <v-file-input
                  :rules="rules"
                  accept="application/pdf"
                  placeholder="Upload Transcript"
                  prepend-icon="mdi-upload"
                  label="Upload PDF Transcript"
                ></v-file-input>
            <v-card-actions>
              <v-btn @click="uploadModal = false">Close</v-btn>
            </v-card-actions>
          </v-card>
        </v-dialog>

      </v-main>
    </v-layout>
  </v-card>
</template>

<script>
export default {
  data() {
    return {
      drawer: true,
      rail: true,
      uploadModal: false,
    };
  }
};
</script>
