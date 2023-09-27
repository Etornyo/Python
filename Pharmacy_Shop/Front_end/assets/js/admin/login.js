const BACKEND_ROOT_URL = "http://127.0.0.1:8000"

const login_form = document.querySelector("form")

login_form,addEventListener("submit", async(event) => {
    event.preventDefault()
    event.stopPropagation()

    const form_data = new FormData(login_form);
    const encoded_data = new URLSearchParams(form_data).toString()

    const result = await fetch(`${BACKEND_ROOT_URL}/admins/authenticate`, {
        method: "POST",
        headers: {
            "Context-Type": "application/x-www-form-urlencoded"
        },
        body: encoded_data
    });
    
    if (result.ok) {
        const token_response = await result.json()
        const access_token = token_response.access_token

        window.localStorage.setItem(user_token, access_token)
        window.Location = "/admin/home.html"
    } 
    else if (result.status === 401) {
        // Handle unauthorized (401) response here
        const alertPlaceholder = document.getElementById('liveAlertPlaceholder');
        const appendAlert = (message, type) => {
            const wrapper = document.createElement('div');
            wrapper.innerHTML = [
                `<div class="alert alert-${type} alert-dismissible" role="alert">`,
                `   <div>${message}</div>`,
                '   <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>',
                '</div>'
            ].join('');
            alertPlaceholder.append(wrapper);
        };
        const error_message = await result.json();
        appendAlert(error_message.detail, 'danger');
    }})