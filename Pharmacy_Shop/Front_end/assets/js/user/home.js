// Get the sidebar element and a reference to the main content
const sidebar = document.getElementById('sidebar');
const mainContent = document.querySelector('.main-content');

// Function to toggle the sidebar between collapsed and expanded state
function toggleSidebar() {
    sidebar.classList.toggle('collapsed');
    mainContent.classList.toggle('expanded');
}

// Add a click event listener to a button or element that triggers the toggle
const toggleButton = document.getElementById('toggle-button'); // Replace with your actual button
toggleButton.addEventListener('click', toggleSidebar);
