function convert() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (file) {
        const resultContainer = document.getElementById('result');
        resultContainer.classList.remove('hidden');

        const pdfViewer = document.getElementById('pdfViewer');
        pdfViewer.src = URL.createObjectURL(file);

        const formData = new FormData();
        formData.append('file', file);

        fetch('/convert', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const jsonViewer = document.getElementById('jsonViewer');
            jsonViewer.textContent = JSON.stringify(data, null, 2);
        })
        .catch(error => console.error('Error:', error));
    } else {
        alert('Please select a PDF file.');
    }
}
