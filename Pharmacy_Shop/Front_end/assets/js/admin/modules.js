const BACKEND_ROOT_URL = "http://127.0.0.1:8000";

export function redirect_to_admin_login() {
    window.location.href = "/admin/login.html";
}

export async function get_current_admin() {
    const admin_token = window.localStorage.getItem("admin_token");

    if (admin_token === null) {
        // Handle the case when admin_token is null (not authenticated)
        redirect_to_admin_login();
    }

    const response = await fetch(`${BACKEND_ROOT_URL}/admin/current`, {
        method: "GET",
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${admin_token}`
        }
    });

    if (response.status === 401) {
        redirect_to_admin_login();
    }

    // Handle other responses here as needed
}

export async function get_inventory_list() {
    const response = await fetch(`${BACKEND_ROOT_URL}/admin/current`, {
        method: "GET",
        headers: {
            "Content-Type": "application/json"
        }
    });

    if (response.ok) {
        return await response.json();
    } else {
        console.log("Failed to load inventory list");
        // Handle the error appropriately
    }

    export async function delete_item_from_inventory(inventory_id){
        const response = await fetch(`${BACKEND_ROOT_URL}`)
    }

    export function get_admin_
}
