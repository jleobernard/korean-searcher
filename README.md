#Idée de l'algo :

T <- Récupérer de la base la liste des phrases contenant tous les mots fixes.
resultats <- []<br />
<ins>POUR CHAQUE</ins> t de T <ins>FAIRE</ins><br />
&nbsp;&nbsp;&nbsp;mmElts <- Multimap (élément de la requête; liste des positions de l'élément de la requête)<br />
&nbsp;&nbsp;&nbsp;<ins>SI</ins> un chemin existe satisfiant la requête <ins>ALORS</ins><br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;resultats <- resultats + t<br />
&nbsp;&nbsp;&nbsp;<ins>FSI</ins><br />
<ins>FIN POUR</ins>