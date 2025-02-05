import dotenv from 'dotenv';
dotenv.config();

import { Client, GatewayIntentBits } from 'discord.js';

const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.MessageContent,
        GatewayIntentBits.GuildMessageReactions,
        GatewayIntentBits.GuildMembers
    ],
});

client.once('ready', () => {
    console.log('Bot is online!!');
});

client.on('messageCreate', async (message) => {
    if (message.content === '!setup') {
        // Prompt user to select a channel
        const channelPrompt = await message.channel.send('Please mention the channel where the role message will be sent:');
        
        const filter = m => m.author.id === message.author.id;
        const channelCollector = message.channel.createMessageCollector({ filter, max: 1 });

        channelCollector.on('collect', async (msg) => {
            const channel = msg.mentions.channels.first();
            if (!channel) {
                await message.channel.send('Invalid channel. Please try again.');
                return channelCollector.stop();
            }

            const roleMessage = await channel.send('React to this message to get a role!');

            const emojiRolePairs = {};
            let collecting = true;

            while (collecting) {
                const rolePrompt = await message.channel.send('Please enter an emoji-role pair in the format ":emoji: role_name" or type "done" to finish:');
                const roleCollector = message.channel.createMessageCollector({ filter, max: 1 });

                roleCollector.on('collect', async (roleMsg) => {
                    if (roleMsg.content.toLowerCase() === 'done') {
                        collecting = false;
                        await message.channel.send('Setup completed!');
                        return roleCollector.stop();
                    }

                    const match = roleMsg.content.match(/^:(.+): (.+)$/);
                    if (match) {
                        const emoji = match[1];
                        const roleName = match[2];
                        emojiRolePairs[emoji] = roleName;
                        await message.channel.send(`Added: ${emoji} for role ${roleName}`);
                        await roleMessage.react(emoji);
                    } else {
                        await message.channel.send('Invalid format. Please try again.');
                    }
                });
            }

            client.on('messageReactionAdd', async (reaction, user) => {
                if (reaction.message.partial) await reaction.message.fetch();
                if (reaction.partial) await reaction.fetch();
                if (user.bot) return;

                if (reaction.message.id === roleMessage.id) {
                    const roleName = emojiRolePairs[reaction.emoji.name];
                    const role = reaction.message.guild.roles.cache.find(role => role.name === roleName);
                    const member = reaction.message.guild.members.cache.get(user.id);
                    if (role && member) {
                        await member.roles.add(role);
                    }
                }
            });
        });
    }
});

client.login(process.env.DISCORD_TOKEN);