document.getElementById("input-text").addEventListener("keyup", function(event) {
  // Number 13 is the "Enter" key on the keyboard
  if (event.keyCode === 13) {
    event.preventDefault();
    document.getElementById("send").click();
  }
});

document.getElementById("send").onclick = async () => {
    const message = document.getElementById("input-text").value;

    fetch(`${window.origin}/predict`, {
        method: "POST",
        credentials: "include",
        body: JSON.stringify({"message": message}),
        cache: "no-cache",
        headers: new Headers({
            "content-type": "application/json"
        })
    })
    .then(function (response) {
        if (response.status !== 200) {
            console.log(`Looks like there was a problem. Status code: ${response.status}`);
            return;
        }
        response.json().then(function (answer) {
            document.getElementById("output-text").innerText = answer;
        });
    })
    .catch(function (error) {
        console.log("Fetch error: " + error);
    });
};
