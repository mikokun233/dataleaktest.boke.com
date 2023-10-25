function createDiamante(noun1, adjective1, verb1, noun2, verb2, adjective2, noun3) {
    let fake_domain = "boke.com"
    let fake_password = "h92ba309ee432bccb91"
    return `${noun1}\n${adjective1} ${verb1}\n${noun1} ${verb1} ${noun2}\n${noun1} ${verb1} ${noun2} ${verb2}\n${noun2} ${verb2} ${adjective2}\n${verb2} ${noun3}\n${noun3}`;
}

console.log(createDiamante('Love', 'Beautiful', 'blossoms', 'Hate', 'decays', 'Ugly', 'Indifference'));
