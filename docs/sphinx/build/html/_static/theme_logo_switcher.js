document.addEventListener("DOMContentLoaded", function() {
    var logo = document.querySelector(".logo img");
    if (!logo) return;

    function updateLogo() {
        var theme = document.documentElement.getAttribute("data-theme");
        var filePath = window.location.pathname;
        var index = filePath.indexOf("/html")
        var result = filePath.substring(0, index + "/html".length) + "/";
        if (theme === "dark") {
            logo.src = result + "_static/PBSP_white_line.png";
        } else {
            logo.src = result + "_static/PBSP_black_line.png";
        }
    }

    // Update logo on page load
    updateLogo();

    // If your theme supports theme switching, listen for changes and update the logo
    var observer = new MutationObserver(updateLogo);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
});
