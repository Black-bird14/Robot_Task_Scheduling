const darkModeToggle=document.getElementById("dark-mode");
const body= document.body;
darkModeToggle.addEventListener('change', ()=>{
    body.classList.toggle('dark');
});