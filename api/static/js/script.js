function addAuthorizedUser() {
    console.log("addAuthorizedUser function called");
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

    fetch("/authorize_user", {
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
        window.location.href = "/";
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
        window.location.href = "/";
    })
    .catch(error => {
        console.error("Error authorizing user:", error);
        alert("An error occurred while adding the authorized user.");
    });
}

function addVisitorUser() {
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

    fetch("/visiter_user", {
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
        window.location.href = "/";
    })
    .catch(error => {
        console.error("Error authorizing user:", error);
        alert("An error occurred while adding the authorized user.");
    });
}

function addUser(event) {
    event.preventDefault();
    console.log("Button clicked");

    fetch("/add_user", {
        method: "GET",
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Server error: " + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        if (data.message && data.message.includes("photo")) {
            window.location.href = "/add_user_verify";
        } else if (data.error) {
            alert("Error: " + data.error);
        }
    })
    .catch(error => {
        console.error("Error:", error);
        alert("An error occurred while processing the request.");
    });
}

function verifyUser(event) {

    event.preventDefault();
    console.log("Button clicked");

    fetch("/verify_user", {
        method: "POST",
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Server error: " + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        if (data.message && data.message.includes("verify photo")) {
            window.location.href = "/verify_user_post";
        } else if (data.error) {
            alert("Error: " + data.error);
        }
    })
    .catch(error => {
        console.error("Error:", error);
        alert("An error occurred while processing the request.");
    });
}

function verifyUserPost(event) {
    fetch('/verify_user_post2', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        if (data) {
            console.log("Server response:", data[0].message);
            alert(data[0].message);

            if (data[0].message.startsWith("Admin user:") || 
                data[0].message.startsWith("Autherised") || 
                data[0].message.startsWith("Visitor")) {

                const admin_user_temp = data[1].data.admin_user_temp;
                const user_type_temp = data[1].data.user_type_temp;
                const details = data[1].data.details;

                console.log(admin_user_temp);
                console.log(user_type_temp);

                fetch('/post_verification', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        admin_user_temp: admin_user_temp,
                        user_type_temp: user_type_temp
                    })
                })
                .then(response => {
                    if (response.ok) {
                        return response.text().then(html => {
                            document.open();
                            document.write(html);
                            document.close();
                        });
                    } else {
                        console.error('Failed to load the page');
                    }
                })
                .catch(error => console.error('Error:', error));
            }
        }
    })
    .catch(error => console.error('Error:', error));
}

function redirectToHome() {
    window.location.href = "/"; 
}

function reset() {
    fetch('/reset', {
        method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
        if (data.message) {
            console.log("Server response:", data.message);
            alert(data.message); 
        }
    })
    .catch(error => console.error('Error:', error));
}


document.getElementById("add-user-btn").addEventListener("click", addUser);
document.getElementById("verify-user-btn").addEventListener("click", verifyUser);
document.getElementById("add-reset-btn").addEventListener("click", reset);