import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Main {

	public static void main(String[] args) throws IOException {
		String caminho = "/home/kranz12/Documents/Projetos/AnaliseSentimento/teste.csv", line = "";
		BufferedReader br = new BufferedReader(new FileReader(caminho));
		
		while ((line = br.readLine()) != null) {
		    String[] row = line.split(",");
		    
		    for(int i = 0;i < row.length;i++){
		    	System.out.println(row[i]);	
		    }
		    System.out.println("------------------------------------");
		}
		br.close();
		
	}

}
