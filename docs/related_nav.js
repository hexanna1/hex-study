(() => {
  const nav = document.querySelector(".related-nav")
  const toggle = nav.querySelector(".related-menu-toggle")
  const links = nav.querySelector(".related-menu-links")

  function setOpen(open) {
    links.hidden = !open
    toggle.setAttribute("aria-expanded", String(open))
  }

  toggle.addEventListener("click", () => setOpen(links.hidden))

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !links.hidden) {
      event.preventDefault()
      event.stopPropagation()
      setOpen(false)
      toggle.focus()
    } else if (nav.contains(event.target)) {
      event.stopPropagation()
    }
  })

  function closeFromOutside(event) {
    if (!links.hidden && !nav.contains(event.target)) setOpen(false)
  }

  document.addEventListener("focusin", closeFromOutside)
  document.addEventListener("click", closeFromOutside)
})()
