function handleTagClick(tag, destination) {
    localStorage.setItem('selectedTag', tag);
    localStorage.setItem('destinationPage', destination);
    window.location.href = '/Digital-garden/' + destination + '/';
}


document.addEventListener('DOMContentLoaded', function() {
    var selectedTag = localStorage.getItem('selectedTag');
    var destinationPage = localStorage.getItem('destinationPage');
    if (selectedTag && destinationPage) {
        var checkboxId = '.js-iso-' + selectedTag; 
        var checkbox = document.getElementById(checkboxId);
        if (checkbox) {
            checkbox.checked = true;
            $('#filters input').change(); // Trigger the change event for the checkboxes
        }
        localStorage.removeItem('selectedTag'); // Clear the stored tag
        localStorage.removeItem('destinationPage'); // Clear the stored destination page
    }
});
