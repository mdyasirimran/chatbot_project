document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const chatLog = document.getElementById('chat-log');
    const modelSelect = document.getElementById('model-select');
    const themeToggle = document.getElementById('theme-toggle'); // Get the new theme toggle button
    const body = document.body; // Reference to the <body> element

    // Function to display a message in the chat log
    // Added 'isTyping' parameter for the typing indicator
    function displayMessage(sender, message, isTyping = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);
        
        // If it's a typing indicator, add the 'typing' class
        if (isTyping) {
            messageDiv.classList.add('typing');
        }
        
        messageDiv.textContent = message;
        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight; // Auto-scroll to bottom

        // Apply animation after element is added to DOM for actual messages
        // Typing indicator has its own animation via CSS
        if (!isTyping) {
            messageDiv.style.opacity = '1';
            messageDiv.style.transform = 'scale(1)';
        }
        return messageDiv; // Return the message element so it can be removed later
    }

    // Function to send message to the Flask API
    async function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return; // Don't send empty messages

        displayMessage('user', message); // Display user's message immediately
        userInput.value = ''; // Clear input field
        userInput.focus(); // Keep focus on input for quick follow-ups

        const selectedModel = modelSelect.value; // Get selected model endpoint

        let typingIndicator = null; // Variable to hold the typing indicator element
        try {
            // Show a typing indicator *before* the fetch request
            typingIndicator = displayMessage('bot', 'Typing...', true);

            const response = await fetch(`/${selectedModel}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(`HTTP error! Status: ${response.status}, Message: ${errorData.response || 'Unknown error'}`);
            }

            const data = await response.json();

            // Remove the "Typing..." indicator before displaying the actual response
            if (typingIndicator) {
                chatLog.removeChild(typingIndicator);
            }

            displayMessage('bot', data.response); // Display bot's actual response

        } catch (error) {
            console.error('Error sending message:', error);
            // Remove the "Typing..." indicator if an error occurred
            if (typingIndicator) {
                chatLog.removeChild(typingIndicator);
            }
            displayMessage('bot', `Error: ${error.message || 'Could not get response from the chatbot.'}`);
        }
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault(); // Prevent default enter behavior (e.g., new line in textarea)
            sendMessage();
        }
    });

    // --- Theme Toggling Logic ---
    function setTheme(theme) {
        if (theme === 'dark') {
            body.classList.add('dark-mode');
            themeToggle.textContent = 'Light Mode'; // Change button text
        } else {
            body.classList.remove('dark-mode');
            themeToggle.textContent = 'Dark Mode'; // Change button text
        }
        localStorage.setItem('theme', theme); // Save preference in local storage
    }

    // Event listener for the theme toggle button
    themeToggle.addEventListener('click', () => {
        const currentTheme = localStorage.getItem('theme');
        // Toggle theme based on current preference
        setTheme(currentTheme === 'dark' ? 'light' : 'dark');
    });

    // Check for saved theme preference on page load
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        setTheme(savedTheme); // Apply saved theme
    } else {
        // Default to light mode if no preference is saved
        setTheme('light');
    }
});