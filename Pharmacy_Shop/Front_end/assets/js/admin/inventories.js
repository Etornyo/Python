import { get_current_admin, get_inventory_list } from "./modules.js";

document.addEventListener("DOMContentLoaded", async (event) => {
    // Ensure the DOM is fully loaded before executing this code

    await get_current_admin();

    const inventory_list = await get_inventory_list();

    const inventory_table = document.querySelector("#inventory-table");

    for (const inventory of inventory_list) {
        const row = document.createElement("tr");

        const nameCell = document.createElement("td");
        nameCell.innerText = inventory.name;

        const priceCell = document.createElement("td");
        priceCell.innerText = inventory.price;

        const quantityCell = document.createElement("td");
        quantityCell.innerText = inventory.quantity;

        const actionCell = document.createElement("td");
        const actionButton = document.createElement("button");
        actionButton.classList.add("btn", "btn-danger");
        actionButton.innerHTML = '<i class="bi bi-trash3"></i>';

        action.addEventListener(DOMContent)

        

        actionCell.appendChild(actionButton);

        row.appendChild(nameCell);
        row.appendChild(priceCell);
        row.appendChild(quantityCell);
        row.appendChild(actionCell);

        inventory_table.appendChild(row)
    }
});

const action = document.createElement(td)


const.addEventListener("click", async (event) =>{
    await delete_item_from_inventory(event.target.id)

    row.remove()
})
