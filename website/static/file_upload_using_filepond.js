FilePond.registerPlugin(FilePondPluginFileValidateSize);

// Turn input element into a pond with configuration options
pond = document.querySelector('.my-pond')
form_root = document.querySelector('.form-root')
var filepond = FilePond.create(
    pond,
    {
	allowMultiple:false,
	maxFileSize: '1MB',
	server: {
	    url: './',
	    process: {
		url:'./upload',  // flask data processing app
		headers:{'X-CSRF-TOKEN': document.querySelector('input[name="csrf_token"]').getAttribute("value")},
		onload: onResponse,
	    },
	},
    },
);

form_root.addEventListener('FilePond:addfile', e=> {
    console.log('File added', e.detail);
});

function onResponse(r){
    
    // create radio button with value of filename right after pond
    r=JSON.parse(r);
    let filename=r.filename[0];
    console.log('filename by onload: '+filename);
    
    file_selection = document.querySelector('.file_selection')
    while(file_selection.firstChild){
	file_selection.removeChild(file_selection.firstChild);
    }
    var radio_button = document.createElement('input')
    radio_button.setAttribute('type', 'radio');
    radio_button.setAttribute('id','radio0');
    radio_button.setAttribute('name','selected_file');
    radio_button.setAttribute('value',filename);
    radio_button.setAttribute('checked',true);
    var label=document.createElement('label');
    label.setAttribute('for','radio0');
    label.appendChild(radio_button);
    label.textContent=filename;
    file_selection.appendChild(radio_button);
    file_selection.appendChild(label);
}
