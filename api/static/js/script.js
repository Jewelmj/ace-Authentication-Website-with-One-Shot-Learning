function addAuthorizedUser() {
    const name = prompt("Enter the name of the user to authorize:");
    if (!name) {
        alert("User name is required!");
        return;
    }

    fetch("/authorize_user", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ name })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(`Error: ${data.error}`);
        } else {
            alert(data.message);
        }
    })
    .catch(error => {
        console.error("Error authorizing user:", error);
        alert("An error occurred while adding the authorized user.");
    });
}

function addAdminUser() {
    const name = prompt("Enter the name of the user to authorize:");
    const major = prompt("Enter the major of the user to authorize:");
    const id_number = prompt("Enter the id_number of the user to authorize:");
    const email = prompt("Enter the email of the user to authorize:");
    const address = prompt("Enter the address of the user to authorize:");
    const phone = prompt("Enter the phone of the user to authorize:");
    const linkedin = prompt("Enter the linkedin of the user to authorize:");
    if (!name) {
        alert("User name is required!");
        return;
    }
    if (!major) {
        alert("User major is required!");
        return;
    }
    if (!id_number) {
        alert("User id_number is required!");
        return;
    }
    if (!email) {
        alert("User email is required!");
        return;
    }
    if (!address) {
        alert("User address is required!");
        return;
    }
    if (!phone) {
        alert("User phone is required!");
        return;
    }
    if (!linkedin) {
        alert("User linkedin is required!");
        return;
    }

    const formData = new FormData();
    formData.append("name", name);
    formData.append("major", major);
    formData.append("id_number", id_number);
    formData.append("email", email);
    formData.append("address", address);
    formData.append("phone", phone);
    formData.append("linkedin", linkedin);

    fetch("/admin_user", {
        method: "POST",
        body: formData 
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(`Error: ${data.error}`);
        } else {
            alert(data.message);
        }
    })
    .catch(error => {
        console.error("Error authorizing user:", error);
        alert("An error occurred while adding the authorized user.");
    });
}

function verifyUser(event) {
    event.preventDefault();
    fetch("/verify_user", {
        method: "POST",
    })
    .then(response => {
        if (!response.ok) {
            return response.text().then((text) => {
                console.error("Server responded with an error:", text);
                throw new Error("Server error: " + response.status);
            });
        }
        return response.json();
    })
    .then(data => {
        if (data.message) {
            console.log("Server response:", data.message);
            alert(data.message);
            if (data.message.startsWith("Admin user:")) {
                window.location.href = "/post_verification";
            }
        }
    })
    .catch(error => {
        console.error("Error verifying user:", error);
        alert("An error occurred while verifying the user.");
    });
}

document.getElementById("add-user-btn").addEventListener("click", addAuthorizedUser);
document.getElementById("add-admin-btn").addEventListener("click", addAdminUser);
document.getElementById("verify-user-btn").addEventListener("click", verifyUser);