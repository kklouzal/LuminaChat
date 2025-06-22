So this is a C++ application with a wxWidgets UI front-end.
D++ (Discord Plus Plus) is used on the backend to interface with discord as a bot.
The main library faciliting the core application functionality is llama.cpp
-----

This is basically a chat application to interface with LLAMA GGUF models.
You can interact with the model directly in the application UI, or over discord.

It supports multiple isolated contexts and a main single shared context.

The main chat interface through the UI hooks into the singular shared context.
You may specify Discord Channel ID's to either be joined to the singular shared context, or be set up to use their own isolated context.
DMing the bot through discord is possible, if enabled through the settings, all DM's use an isolated context.
-----

I've been working on this for a while now, trying to flesh out the basic features, and some more advanced things.
We've got the ability to pull message histories from Discord channels the bot joins, then those get fed into the context so the LLM has context of recent chat history.
Contexts are automatically pruned and managed, pruned messages get summarized and injected back into the back of the context to maintain as much histrical chat detail as possible.
-----

The LLM generally has the ability to distinguish and interact in a multi-user chat environment, my test environment is using a custom chat template which I'll publish here eventually.
The custom chat template has some small adjustments from the normal LLAMA template to help with multi-user chat environments.
-----

Let's see.. What else... This project has a lot going on inside of it. I try not to commit to master unless I've extensively tested every current feature and functionality.
I tend to have a lot of branches at once, I clean them up once I get to a milestone and merge everything back down into master.
-----

Ther are plenty of features I haven't acknowledged here, I'll update this as time permits to give a more concise overview of what's available.
-----

Q: Did you use AI to code this?
A: Yep. I've been working with c++ for over 20 years. AI programming assistants have come along very far over the last couple years and they're capable of managing a project of this size now.
I step in and make manual adjustments where I need to, but the AI programming assistant lets me focus on core design and logic. It's accelerated the development of this project probably over 100X

Q: Why did you make this?
A: I was interested in what I could do with C++ to interface with LLM's. I found llama.cpp and began plugging away. I then started playing with the AI programming assistant within my toolchains and that lead us to where we are now.
So this was mainly fueled by curiosity in a few different domains, and now has turned into a bit of a passion project.

Q: Are you done adding more features?
A: Nope, I have plenty more planned. The development cycle consists of:
  1. Add new feature.
  2. Test new feature.
  3. Eliminate bugs.
  4. Optimize & cleanup code from new feature.
  5. Test optimized feature code.
  6. Eliminate bugs.
  8. Test new feature again.
  9. Final code cleanup & optimization run.
  10. Final test for feature completeness and stability.
  11. Push to master.
So the development cycle is a bit crazy and can get pretty complicated at times, especially with using the AI coding assistant.

Q: Where do you expect this project to end up?
A: No idea, I have no end in sight and continue to chug along.
